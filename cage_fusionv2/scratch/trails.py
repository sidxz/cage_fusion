from transformers import AutoTokenizer

model_ckpt = "DeepChem/ChemBERTa-77M-MTR"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(f"Loaded tokenizer from {model_ckpt}")
print("Tokenizer vocab size:", tokenizer.vocab_size)
print("Model max length:", tokenizer.model_max_length)
print("Tokenizer special tokens:", tokenizer.special_tokens_map)


test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
inputs = tokenizer(test_smiles, return_tensors="pt", padding=True, truncation=True)
print(inputs)


# Load existing model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
from transformers import AutoModel
model = AutoModel.from_pretrained(model_ckpt).to(device)
print(f"Loaded model from {model_ckpt}")

# Test a forward pass
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
print("Model output keys:", outputs.keys())
print("Last hidden state shape:", last_hidden_state.shape)

# Return vector only for the [CLS] token
cls_vectors = last_hidden_state[:, 0, :]
print("CLS token vectors shape:", cls_vectors.shape)




