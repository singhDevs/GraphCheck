from model.llm import LLM
from model.graph_llm import GraphLLM


load_model = {
    "llm": LLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

