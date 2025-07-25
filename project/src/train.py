import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model.transformer import SpikingMoELLM, SpikingLLM
import torch.nn.functional as F
from snntorch import utils
from tqdm import tqdm
from huggingface_hub import login, HfApi
from datetime import date, datetime
import os, json
#import wandb
#wandb.init(
#    project="spiking-llm-training-local",
#)
login("hf_XXX")

REPO_NAME = f"Chan-Y/spiking-llm-{date.today()}-{datetime.now()}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=512):
        self.samples = []
        for text in texts:
            tokens = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
            self.samples.append(tokens["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return ids, ids  # input_ids, labels

vocab_size = tokenizer.vocab_size

moe_config = {
    'vocab_size': vocab_size+1,
    'd_model': 2048,
    'n_heads': 16,
    'n_kv_heads': 8,
    'num_layers': 16,
    'max_seq_len': 512,
    'beta': 0.95,
    'num_experts': 2,
    'num_active': 1,
    'device': DEVICE,
    'dtype': torch.bfloat16
}

dense_config = {
    'vocab_size': vocab_size+1,
    'd_model': 2048,
    'n_heads': 16,
    'n_kv_heads': 8,
    'num_layers': 8,
    'max_seq_len': 512,
    'beta': 0.95,
    'device': DEVICE,
    'dtype': torch.bfloat16
}

dense_config_400m = {
    'vocab_size': vocab_size+1,
    'd_model': 1024,
    'n_heads': 16,
    'n_kv_heads': 8,        
    'num_layers': 10,       
    'max_seq_len': 512,
    'beta': 0.95,
    'device': DEVICE,
    'dtype': torch.float32
}

#model = SpikingMoELLM(**moe_config).to(DEVICE)
model = SpikingLLM(**dense_config_400m).to(DEVICE)

#model = torch.compile(model)

texts = open("taylorswift.txt").read()
dataset = TextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

scaler = torch.amp.GradScaler('cuda')

print(f"Dataset size: {len(dataset)} samples")
print(f"Model parameters: {model.get_num_params():,}")

#wandb.log({
#    'model_parameters': model.get_num_params(),
#    'vocab_size': vocab_size,
#    'dataset_size': len(dataset)
#})

model.train()
progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")

total_loss = 0
log_interval = 10

for step, (input_ids, labels) in progress_bar:
    input_ids = input_ids.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    utils.reset(model) 

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss.item()

    #wandb.log({
    #    'step': step,
    #    'loss': loss.item(),
    #    'learning_rate': optimizer.param_groups[0]['lr']
    #})

    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
print("Training completed successfully!")

save_dir = './saved_model'
os.makedirs(save_dir, exist_ok=True)

print("Saving model and tokenizer")
torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
# Save model config
config_dict = dense_config.copy()
config_dict['device'] = str(config_dict['device'])  # Convert device to string for JSON
config_dict['dtype'] = str(config_dict['dtype'])    # Convert dtype to string for JSON

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

# Save tokenizer
tokenizer.save_pretrained(save_dir)

# Log final metrics
final_avg_loss = total_loss / len(loader)
#wandb.log({
#    "final_average_loss": final_avg_loss,
#    "total_steps": len(loader)
#})

print(f"Final average loss: {final_avg_loss:.4f}")

# Push to Hugging Face
print("Pushing model and tokenizer to Hugging Face...")

try:
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_NAME, exist_ok=True)
        print(f"Repository {REPO_NAME} created/verified")
    except Exception as e:
        print(f"Repository creation info: {e}")
    
    # Upload all files in the save directory
    api.upload_folder(
        folder_path=save_dir,
        repo_id=REPO_NAME,
        commit_message="Upload trained spiking LLM model"
    )
    
    # Create a model card
    model_card = f"""---
license: mit
library_name: pytorch
tags:
- spiking-neural-networks
- llm
- language-model
- pytorch
---

# Spiking LLM Model

This is a Spiking Neural Network based Language Model trained on Taylor Swift data.

## Model Details

- Model Type: SpikingLLM
- Parameters: {model.get_num_params():,}
- Architecture: {dense_config['num_layers']} layers, {dense_config['d_model']} hidden size
- Vocabulary Size: {vocab_size}
- Training Loss: {final_avg_loss:.4f}

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

# Load model (you'll need the custom model class)
model_state = torch.load("pytorch_model.bin")
# Initialize your SpikingLLM with the config and load the state
```

## Training Details

- Optimizer: AdamW
- Learning Rate: 5e-4
- Batch Size: 1
- Sequence Length: 512

"""
    
    # Upload model card
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_NAME,
        commit_message="Add model card"
    )
    
    print(f"✅ Model successfully uploaded to: https://huggingface.co/{REPO_NAME}")
        
except Exception as e:
    print(f"❌ Error uploading to Hugging Face: {e}")

#wandb.finish()

print("Training and upload process completed!")
