import os
import json
import random
from pathlib import Path
from typing import Iterator, List, Dict, Any
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoTokenizer
import pyarrow.parquet as pq

class SimpleFineWebPreprocessor:
    def __init__(self, 
                 data_dir: str = "./fineweb2/",
                 output_dir: str = "./processed_data/",
                 tokenizer_name: str = "google/gemma-2-2b",  # Gemma-3 tokenizer
                 max_seq_length: int = 2048,
                 train_split: float = 0.95,
                 use_fraction: float = 0.25,  # Use 1/4 of dataset
                 chunk_size: int = 10000):  # Save data in chunks
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_seq_length = max_seq_length
        self.train_split = train_split
        self.use_fraction = use_fraction
        self.chunk_size = chunk_size
        
        # Load Gemma tokenizer
        print(f"Loading Gemma tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Gemma uses different special tokens
        print(f"Gemma vocab size: {self.tokenizer.vocab_size}")
        print(f"BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        print(f"EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir / "tokenizer")
        
        # Stats tracking
        self.total_docs = 0
        self.total_tokens = 0
        self.total_sequences = 0
    
    def find_parquet_files(self) -> List[Path]:
        """Find all Turkish parquet files"""
        pattern = "data/tur_Latn/train/*.parquet"
        parquet_files = list(self.data_dir.glob(pattern))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found with pattern: {self.data_dir}/{pattern}")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Use only a fraction for 1/4 dataset
        num_files_to_use = max(1, int(len(parquet_files) * self.use_fraction))
        selected_files = random.sample(parquet_files, num_files_to_use)
        selected_files.sort()  # Sort for reproducibility
        
        print(f"Using {num_files_to_use} files ({self.use_fraction:.1%} of available data)")
        return selected_files
    
    def load_documents(self, parquet_files: List[Path]) -> Iterator[str]:
        """Load documents from parquet files"""
        for file_path in tqdm(parquet_files, desc="Loading files"):
            try:
                # Read parquet file
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                # Yield just the text content
                for text in df['text']:
                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        yield text.strip()
                        self.total_docs += 1
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    def tokenize_and_chunk(self, text: str) -> List[List[int]]:
        """Tokenize text and create training sequences"""
        # Tokenize the text
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,  # Add BOS/EOS tokens
            truncation=False,
            padding=False
        )
        
        self.total_tokens += len(tokens)
        
        # Create sequences of max_seq_length
        sequences = []
        
        if len(tokens) <= self.max_seq_length:
            # Short text: pad to max_seq_length
            padded = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(tokens))
            sequences.append(padded)
        else:
            # Long text: split with slight overlap
            overlap = 128  # Small overlap to maintain context
            step = self.max_seq_length - overlap
            
            for i in range(0, len(tokens) - self.max_seq_length + 1, step):
                sequence = tokens[i:i + self.max_seq_length]
                sequences.append(sequence)
        
        return sequences
    
    def save_chunk(self, sequences: List[List[int]], chunk_idx: int, split: str):
        """Save a chunk of sequences"""
        chunk_path = self.output_dir / f"{split}_chunk_{chunk_idx:04d}.pt"
        
        # Convert to tensor
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        
        # Save
        torch.save({
            'input_ids': sequences_tensor,
            'num_sequences': len(sequences),
            'seq_length': self.max_seq_length,
            'chunk_idx': chunk_idx
        }, chunk_path)
        
        print(f"Saved {len(sequences)} sequences to {chunk_path}")
    
    def process_dataset(self):
        """Main processing function"""
        print("="*60)
        print("FINEWEB-2 TURKISH DATA PREPROCESSING (SIMPLE)")
        print("="*60)
        
        # Find files
        parquet_files = self.find_parquet_files()
        
        # Process documents
        all_sequences = []
        chunk_idx = 0
        
        print(f"\nProcessing documents...")
        print(f"Target sequence length: {self.max_seq_length}")
        print(f"Chunk size: {self.chunk_size} sequences")
        
        for text in self.load_documents(parquet_files):
            # Tokenize and create sequences
            sequences = self.tokenize_and_chunk(text)
            all_sequences.extend(sequences)
            self.total_sequences += len(sequences)
            
            # Save chunk when we have enough sequences
            if len(all_sequences) >= self.chunk_size:
                self.save_chunk(all_sequences, chunk_idx, "train")
                all_sequences = []
                chunk_idx += 1
                
                # Print progress
                if chunk_idx % 10 == 0:
                    print(f"Processed {chunk_idx} chunks, {self.total_docs:,} docs, {self.total_sequences:,} sequences")
        
        # Save remaining sequences
        if all_sequences:
            self.save_chunk(all_sequences, chunk_idx, "train")
            chunk_idx += 1
        
        # Create train/validation split by moving some chunks to validation
        self._create_train_val_split(chunk_idx)
        
        # Save metadata
        self._save_metadata(chunk_idx)
        
        print("="*60)
        print("PREPROCESSING COMPLETE!")
        print(f"Total documents: {self.total_docs:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Total sequences: {self.total_sequences:,}")
        print(f"Saved in {chunk_idx} chunks")
        print(f"Data saved to: {self.output_dir}")
        print("="*60)
    
    def _create_train_val_split(self, total_chunks: int):
        """Move some chunks to validation set"""
        num_val_chunks = max(1, int(total_chunks * (1 - self.train_split)))
        
        # Select random chunks for validation
        val_chunk_indices = random.sample(range(total_chunks), num_val_chunks)
        
        print(f"\nCreating train/val split:")
        print(f"Moving {num_val_chunks} chunks to validation")
        
        for chunk_idx in val_chunk_indices:
            old_path = self.output_dir / f"train_chunk_{chunk_idx:04d}.pt"
            new_path = self.output_dir / f"val_chunk_{chunk_idx:04d}.pt"
            
            if old_path.exists():
                old_path.rename(new_path)
        
        train_chunks = total_chunks - num_val_chunks
        print(f"Train chunks: {train_chunks}")
        print(f"Validation chunks: {num_val_chunks}")
    
    def _save_metadata(self, total_chunks: int):
        """Save dataset metadata"""
        # Count actual train/val chunks
        train_chunks = len(list(self.output_dir.glob("train_chunk_*.pt")))
        val_chunks = len(list(self.output_dir.glob("val_chunk_*.pt")))
        
        metadata = {
            'tokenizer_name': 'google/gemma-2-2b',
            'vocab_size': self.tokenizer.vocab_size,
            'max_seq_length': self.max_seq_length,
            'total_documents': self.total_docs,
            'total_tokens': self.total_tokens,
            'total_sequences': self.total_sequences,
            'train_chunks': train_chunks,
            'val_chunks': val_chunks,
            'chunk_size': self.chunk_size,
            'use_fraction': self.use_fraction,
            'special_tokens': {
                'bos_token_id': self.tokenizer.bos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        }
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved to {self.output_dir}/metadata.json")

# Dataset class for training
class FineWebDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading processed FineWeb data"""
    
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Find all chunk files for this split
        self.chunk_files = sorted(self.data_dir.glob(f"{split}_chunk_*.pt"))
        
        if not self.chunk_files:
            raise ValueError(f"No {split} chunk files found in {data_dir}")
        
        print(f"Found {len(self.chunk_files)} {split} chunks")
        
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Calculate total sequences
        self.total_sequences = 0
        self.chunk_sizes = []
        
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location='cpu')
            chunk_size = chunk_data['num_sequences']
            self.chunk_sizes.append(chunk_size)
            self.total_sequences += chunk_size
        
        print(f"Total {split} sequences: {self.total_sequences:,}")
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Find which chunk contains this index
        chunk_idx = 0
        sequences_seen = 0
        
        for i, chunk_size in enumerate(self.chunk_sizes):
            if sequences_seen + chunk_size > idx:
                chunk_idx = i
                local_idx = idx - sequences_seen
                break
            sequences_seen += chunk_size
        
        # Load chunk if needed (implement caching if memory allows)
        chunk_data = torch.load(self.chunk_files[chunk_idx], map_location='cpu')
        sequence = chunk_data['input_ids'][local_idx]
        
        return {
            'input_ids': sequence,
            'labels': sequence.clone()  # For causal LM, labels = input_ids
        }

# Usage example
def main():
    # Initialize preprocessor
    preprocessor = SimpleFineWebPreprocessor(
        data_dir="./fineweb2/",
        output_dir="./processed_turkish_data/",
        tokenizer_name="google/gemma-2-2b",
        max_seq_length=2048,
        use_fraction=0.25,  # Use 1/4 of the data
        chunk_size=10000    # 10k sequences per chunk
    )
    
    # Process the dataset
    preprocessor.process_dataset()
    
    # Test loading the dataset
    print("\nTesting dataset loading...")
    
    train_dataset = FineWebDataset("./processed_turkish_data/", split="train")
    val_dataset = FineWebDataset("./processed_turkish_data/", split="val")
    
    print(f"Train dataset size: {len(train_dataset):,}")
    print(f"Val dataset size: {len(val_dataset):,}")
    
    # Test a sample
    sample = train_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    
    print("\n✅ Data preprocessing complete!")

if __name__ == "__main__":
    main()