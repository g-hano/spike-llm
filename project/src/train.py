import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        login("hf_XXX")
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
            return ids, ids

    vocab_size = tokenizer.vocab_size

    # Use a smaller config to avoid memory issues
    config = {
        'vocab_size': vocab_size+1,
        'd_model': 2816,
        'n_heads': 24,
        'n_kv_heads': 8,
        'num_layers': 32,
        'max_seq_len': 2048,
        'beta': 0.95,
        'device': DEVICE,
        'dtype': torch.bfloat16
    }

    model = SpikingLLM(**config).to(DEVICE)
    print(f"Model parameters: {model.get_num_params():,}")
    print(torch.cuda.memory_allocated() / (1024**3))
    
    texts = open("project/src/taylorswift.txt").read()
    dataset = TextDataset(texts, tokenizer, seq_len=1024)
    
    loader = DataLoader(
        dataset, 
        batch_size=1,  # Process one sample at a time to eliminate any graph conflicts
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8
    )


    model.train()
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    total_loss = 0
    effective_batch_size = 32  # We'll simulate this through accumulation
    accumulation_count = 0

    for step, (input_ids, labels) in progress_bar:
        
        # Fresh start for each sample
        if accumulation_count == 0:
            optimizer.zero_grad()
        
        # Reset all memory states to fresh tensors (not in-place!)
        model.reset_mem()
        
        # Create completely fresh tensor copies
        input_ids = torch.tensor(input_ids.numpy(), device=DEVICE, dtype=torch.long)
        labels = torch.tensor(labels.numpy(), device=DEVICE, dtype=torch.long)

        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / effective_batch_size  # Scale for accumulation
        
        # Backward pass - NO retain_graph!
        loss.backward()
        
        accumulation_count += 1
        total_loss += loss.item() * effective_batch_size
        
        # Update when we've accumulated enough gradients
        if accumulation_count >= effective_batch_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            accumulation_count = 0
            
            # Complete cleanup after optimizer step
            model.reset_mem()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Immediate tensor cleanup
        del logits, loss, input_ids, labels
        
        progress_bar.set_postfix({
            "Loss": f"{(total_loss / (step + 1)):.4f}",
            "Acc": f"{accumulation_count}/{effective_batch_size}",
        })
        
        # Periodic deep cleanup
        if step % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final optimizer step if needed
    if accumulation_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    print("✓ Training completed!")

    # Save model
    save_dir = './saved_model'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))

    config_dict = config.copy()
    config_dict['device'] = str(config_dict['device'])
    config_dict['dtype'] = str(config_dict['dtype'])

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    tokenizer.save_pretrained(save_dir)

    final_avg_loss = total_loss / len(loader)
    print(f"Final average loss: {final_avg_loss:.4f}")

    print("🎉 Training completed with proper memory management!")
