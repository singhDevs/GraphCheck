import contextlib
import torch
import warnings
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from src.ckpt import visualize_graph


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        
        num_devices = torch.cuda.device_count()
        
        max_memory = {}
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
            max_memory[i] = f"{max(total_memory - 2, 2)}GiB"
            
        kwargs.update({
            "max_memory": max_memory,
            "device_map": "auto",
            "revision": "main",
        })
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        
        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            
        model.gradient_checkpointing_enable()

        self.model = model
        
        print("After loading model:")
        print(torch.cuda.memory_summary())
        
        print('Finish loading LLAMA!')

        self.word_embedding = self.model.model.get_input_embeddings()

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)
        
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.weight.shape[1]),
        ).to(self.model.device)
        
        self.embed_dim = self.word_embedding.weight.shape[1]
        self.gnn_output = args.gnn_hidden_dim
        

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, data):
        claim_kg = data['claim_kg'].to(self.model.device)
        doc_kg = data['doc_kg'].to(self.model.device)
        
        claim_n_embeds, _ = self.graph_encoder(claim_kg.x, claim_kg.edge_index.long(), claim_kg.edge_attr)
        
        doc_n_embeds, _ = self.graph_encoder(doc_kg.x, doc_kg.edge_index.long(), doc_kg.edge_attr)
        
        # visualization
        index = data['index']
        dataset = data['dataset']
        
        
        if claim_kg.batch is not None:  
            claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')  
        else:  
            claim_embeds = claim_n_embeds.mean(dim=0, keepdim=True)
            
        if doc_kg.batch is not None:  
            doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')  
        else:  
            doc_embeds = doc_n_embeds.mean(dim=0, keepdim=True)

        # mean pooling
        # claim_embeds = scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')
        # doc_embeds = scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')
        return claim_embeds, doc_embeds

    def forward(self, data):

        # texts and labels
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        # descriptions = self.tokenizer(data["desc"], add_special_tokens=False)
        labels = self.tokenizer(data["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].cuda())
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).cuda()).unsqueeze(0)

        # encode graphs
        claim_embeds, doc_embeds = self.encode_graphs(data)
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)

        batch_size = len(data['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)

            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
                
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, data):

        # encode description and questions
        texts = self.tokenizer(data["text"], add_special_tokens=False)
        # descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].cuda())
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).cuda()).unsqueeze(0)

        # encode graphs
        claim_embeds, doc_embeds = self.encode_graphs(data)
        claim_embeds = self.projector(claim_embeds)
        doc_embeds = self.projector(doc_embeds)
        
        data['id'] = [data['id']] if isinstance(data['id'], int) else data['id']
        batch_size = len(data['id'])

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # if claim_embeds or doc_embeds is null
            if claim_embeds.size(0) == batch_size:
                claim_embedding = claim_embeds[i].unsqueeze(0)
            else:
                claim_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)

            if doc_embeds.size(0) == batch_size:
                doc_embedding = doc_embeds[i].unsqueeze(0)
            else:
                doc_embedding = torch.zeros(self.embed_dim).unsqueeze(0).to(self.model.device)
            
            inputs_embeds = torch.cat([bos_embeds, claim_embedding, doc_embedding, inputs_embeds], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True,  # IMPORTANT!
                # eos_token_id=2,
                # pad_token_id=2
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return {'id': data['id'],
                'pred': pred,
                'label': data['label'],
                'text': data['text']}

    def inference_demo(self, data):
        # Encode text
        texts = self.tokenizer(data["text"], add_special_tokens=False)

        # Encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device)
        )
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # Encode graphs
        claim_embeds, doc_embeds = self.encode_graphs(data)
        claim_embeds = self.projector(claim_embeds) if claim_embeds is not None else torch.zeros(self.embed_dim).to(self.model.device)
        doc_embeds = self.projector(doc_embeds) if doc_embeds is not None else torch.zeros(self.embed_dim).to(self.model.device)

        # Process text input
        input_ids = texts.input_ids[:self.max_txt_len] + eos_user_tokens.input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))

        # Concatenate embeddings
        inputs_embeds = torch.cat([bos_embeds, claim_embeds, doc_embeds, inputs_embeds], dim=0)

        # Create attention mask
        attention_mask = torch.tensor([1] * inputs_embeds.shape[0], dtype=torch.long).to(self.model.device)

        # Pad inputs_embeds to max length
        max_length = inputs_embeds.shape[0]
        pad_length = self.max_txt_len + 3 - max_length  # Adjust padding based on max length
        if pad_length > 0:
            inputs_embeds = torch.cat([pad_embeds.repeat(pad_length, 1), inputs_embeds], dim=0)
            attention_mask = torch.cat([torch.zeros(pad_length, dtype=torch.long).to(self.model.device), attention_mask])

        # Generate output
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds.unsqueeze(0),  # Add batch dimension
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask.unsqueeze(0),  # Add batch dimension
                use_cache=True,
            )

        # Decode and return prediction
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(pred)
        return {
            'id': data['id'],
            'pred': pred,
            'label': data['label'],
            'text': data['text']
        }


    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
