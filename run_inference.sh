# List of datasets
datasets=("AggreFact-CNN" "AggreFact-XSum" "summeval" "ExpertQA" "COVID-Fact" "pubhealth" "SCIFACT")

for dataset in "${datasets[@]}"; do
    echo "Running inference for dataset: $dataset"
    python inference.py --dataset_name $dataset
done
