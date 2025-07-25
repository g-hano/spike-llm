import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model.transformer import SpikingMoELLM, SpikingLLM
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import login, HfApi
from datetime import date, datetime
import os, json
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        login("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")
    except:
        print("HF login failed")

    REPO_NAME = f"Chan-Y/spiking-llm-{date.today()}-{datetime.now()}"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Clean config without any memory tricks - just good settings
    dense_config_400m = {
        'vocab_size': vocab_size+1,
        'd_model': 1024,
        'n_heads': 16,
        'n_kv_heads': 8,        
        'num_layers': 10,       
        'max_seq_len': 512,
        'beta': 0.95,
        'device': DEVICE,
        'dtype': torch.bfloat16
    }

    print("Creating model...")
    model = SpikingLLM(**dense_config_400m).to(DEVICE)
    
    # Enable torch.compile for speed (optional - comment out if issues)
    #try:
    #    model = torch.compile(model, mode='default')
    #    print("✓ Torch compile enabled")
    #except Exception as e:
    #    print(f"⚠️ Torch compile failed, continuing without: {e}")

    # Load data
    print("Loading dataset...")
    texts = open("project/src/taylorswift.txt").read()
    dataset = TextDataset(texts, tokenizer)
    
    # Simple, reliable DataLoader for Windows
    loader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Optimizer with good defaults for LLMs
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Simple cosine scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader), eta_min=1e-6)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Training batches: {len(loader)}")
    # Training settings
    accumulation_steps = 8  # Effective batch size = 4 * 8 = 32
    max_grad_norm = 1.0

    model.train()
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    total_loss = 0
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    for step, (input_ids, labels) in progress_bar:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # Reset spiking neuron memory states
        model.reset_mem()

        # Forward pass with mixed precision (bfloat16 - no scaler needed)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / accumulation_steps  # Scale for accumulation
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients and step
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Update progress
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            "Loss": f"{loss.item() * accumulation_steps:.4f}",
            "LR": f"{current_lr:.2e}",
            "Step": f"{step+1}/{len(loader)}"
        })
        
        # Periodic memory cleanup
        if step % 1000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final optimizer step if needed
    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
    print("\n✓ Training completed successfully!")

    # Save model
    save_dir = './saved_model'
    os.makedirs(save_dir, exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))

    # Save config
    config_dict = dense_config_400m.copy()
    config_dict['device'] = str(config_dict['device'])
    config_dict['dtype'] = str(config_dict['dtype'])

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Calculate final metrics
    final_avg_loss = total_loss / len(loader)
    print(f"Final average loss: {final_avg_loss:.4f}")

    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    try:
        api = HfApi()
        api.create_repo(repo_id=REPO_NAME, exist_ok=True)
        print(f"Repository created: {REPO_NAME}")
        
        api.upload_folder(
            folder_path=save_dir,
            repo_id=REPO_NAME,
            commit_message="Upload spiking LLM model (no gradient checkpointing)"
        )
        
        # Create model card
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

A Spiking Neural Network based Language Model trained without gradient checkpointing.

## Model Details

- **Model Type**: SpikingLLM
- **Parameters**: {model.get_num_params():,}
- **Architecture**: {dense_config_400m['num_layers']} layers, {dense_config_400m['d_model']} hidden size
- **Vocabulary Size**: {vocab_size:,}
- **Training Loss**: {final_avg_loss:.4f}

## Training Details

- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.1)
- **Scheduler**: CosineAnnealingLR
- **Batch Size**: 4 (effective: 32 with gradient accumulation)
- **Sequence Length**: 512
- **Mixed Precision**: bfloat16
- **Gradient Checkpointing**: Disabled (incompatible with SNNs)

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

# Load model weights
model_state = torch.load("pytorch_model.bin")
```

## Notes

This model uses spiking neural networks which maintain internal memory states. 
Gradient checkpointing is disabled as it's incompatible with stateful neurons.
"""
        
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=REPO_NAME,
            commit_message="Add model card"
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{REPO_NAME}")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("Model saved locally in ./saved_model/")

    print("\n🎉 Training and upload process completed!")
    
    if torch.cuda.is_available():
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
