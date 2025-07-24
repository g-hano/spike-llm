import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model.transformer import SpikingMoELLM  # adjust if needed
import torch.nn.functional as F
from snntorch import utils
from tqdm import tqdm

TEXT_DIR = r"C:\Users\Cihan\Downloads\text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

# Simple Dataset
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

# Model Init
model = SpikingMoELLM(
    vocab_size=vocab_size+1,
    d_model=2048,
    n_heads=16,
    n_kv_heads=8,
    num_layers=16,
    max_seq_len=512,
    beta=0.95,
    num_experts=2,
    num_active=1,
    device=DEVICE,
    dtype=torch.float32
).to(DEVICE)

# Data
texts = open(r"C:\Users\Cihan\Desktop\snn\project\data\taylorswift.txt").read()
dataset = TextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print(f"Dataset size: {len(dataset)} samples")
print(f"Model parameters: {model.get_num_params():,}")

# Training loop (1 epoch)
model.train()
progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")

for step, (input_ids, labels) in progress_bar:
    input_ids = input_ids.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    utils.reset(model)  # 🔧 RESET SPIKING NEURONS HERE!

    logits, _ = model(input_ids)  # Fixed syntax error
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    # Update progress bar with current loss
    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    

print("Training completed successfully!")