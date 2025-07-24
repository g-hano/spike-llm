import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model.transformer import SpikingMoELLM  # adjust if needed
import torch.nn.functional as F
from snntorch import utils
from tqdm import tqdm

TEXT_DIR = r"C:\Users\Cihan\Downloads\text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-GPU setup for Kaggle
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")
    DEVICE = "cuda"
else:
    n_gpus = 0
    DEVICE = "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

# Simple Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=512):
        self.samples = []
        # If texts is a single string, split it into chunks
        if isinstance(texts, str):
            # Split into chunks for better training
            chunk_size = seq_len * 2  # Overlap chunks for more data
            text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            texts = text_chunks
        
        for text in texts:
            if len(text.strip()) > 0:  # Skip empty texts
                tokens = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
                self.samples.append(tokens["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return ids, ids  # input_ids, labels

vocab_size = tokenizer.vocab_size

# Model Init
model = SpikingMoELLM(
    vocab_size=vocab_size+1,
    d_model=512,
    n_heads=8,
    n_kv_heads=4,
    num_layers=4,
    max_seq_len=512,
    beta=0.95,
    num_experts=4,
    num_active=2,
    device=DEVICE,
    dtype=torch.float32
).to(DEVICE)

# Multi-GPU setup
if n_gpus > 1:
    print(f"Using {n_gpus} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)
    # Increase batch size to utilize multiple GPUs better
    batch_size = 2 * n_gpus
else:
    batch_size = 2

# Data - Fixed text loading
with open(r"project\data\taylorswift.txt", 'r', encoding='utf-8') as f:
    texts = f.read()

dataset = TextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"Dataset size: {len(dataset)} samples")
print(f"Batch size: {batch_size}")
if hasattr(model, 'module'):  # DataParallel wrapper
    print(f"Model parameters: {model.module.get_num_params():,}")
else:
    print(f"Model parameters: {model.get_num_params():,}")

# Training loop (1 epoch)
model.train()
progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")

for step, (input_ids, labels) in progress_bar:
    input_ids = input_ids.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    optimizer.zero_grad()

    # Reset spiking neurons - handle DataParallel
    if hasattr(model, 'module'):
        utils.reset(model.module)  # Access the actual model inside DataParallel
    else:
        utils.reset(model)

    logits, _ = model(input_ids)
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    # Update progress bar with current loss and GPU utilization
    if n_gpus > 1:
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "GPUs": n_gpus,
            "GPU_Mem": f"{gpu_mem:.1f}GB"
        })
    else:
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    

print("Training completed successfully!")

# GPU memory summary
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} max memory used: {torch.cuda.max_memory_allocated(i) / 1024**3:.1f}GB")
    torch.cuda.empty_cache()  # Clear GPU cache
