import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from datasets import load_from_disk, Dataset

class HuggingFaceTokenizationPipeline:
    """Optimized tokenization pipeline for HuggingFace datasets"""
    
    def __init__(self, tokenizer_path: str, max_length: int = 2048, num_workers: int = None):
        """
        Args:
            tokenizer_path: Path to trained BPE tokenizer
            max_length: Maximum sequence length
            num_workers: Number of parallel workers
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        self.num_workers = num_workers or mp.cpu_count()
        
        # Configure tokenizer for efficiency
        self.tokenizer.enable_padding(length=max_length, pad_token="<pad>")
        self.tokenizer.enable_truncation(max_length=max_length)
        
        print(f"✅ Initialized tokenizer with {self.num_workers} workers")
    
    def load_hf_dataset(self, dataset_path: str) -> Dataset:
        """Load HuggingFace dataset and verify structure"""
        print(f"📁 Loading dataset from {dataset_path}...")
        
        try:
            dataset = load_from_disk(dataset_path)
            print(f"  ✅ Loaded {len(dataset):,} samples")
            print(f"  ✅ Columns: {dataset.column_names}")
            
            # Verify text column exists
            text_columns = [col for col in dataset.column_names 
                          if any(keyword in col.lower() for keyword in ['text', 'content', 'body'])]
            
            if not text_columns:
                raise ValueError(f"No text column found in dataset. Available columns: {dataset.column_names}")
            
            text_column = text_columns[0]
            print(f"  ✅ Using text column: '{text_column}'")
            
            # Show sample
            if len(dataset) > 0:
                sample_text = dataset[0][text_column]
                print(f"  📝 Sample text: {str(sample_text)[:100]}...")
            
            return dataset, text_column
            
        except Exception as e:
            print(f"  ❌ Error loading dataset: {e}")
            raise
    
    def tokenize_batch_hf(self, batch: Dict[str, List], text_column: str) -> Dict[str, List]:
        """Tokenize a batch from HuggingFace dataset"""
        texts = batch[text_column]
        
        # Filter valid texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str):
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            return {
                'input_ids': [],
                'length': [],
                'attention_mask': []
            }
        
        # Batch tokenize (much faster than individual)
        try:
            encodings = self.tokenizer.encode_batch(valid_texts)
            
            # Prepare outputs for all samples (including filtered ones)
            all_input_ids = [None] * len(texts)
            all_lengths = [0] * len(texts)
            all_attention_masks = [None] * len(texts)
            
            # Fill in successful tokenizations
            for i, encoding in enumerate(encodings):
                original_idx = valid_indices[i]
                all_input_ids[original_idx] = encoding.ids
                all_lengths[original_idx] = len(encoding.ids)
                all_attention_masks[original_idx] = encoding.attention_mask
            
            # Filter out None values (failed tokenizations)
            final_input_ids = [ids for ids in all_input_ids if ids is not None]
            final_lengths = [length for length in all_lengths if length > 0]
            final_attention_masks = [mask for mask in all_attention_masks if mask is not None]
            
            return {
                'input_ids': final_input_ids,
                'length': final_lengths,
                'attention_mask': final_attention_masks
            }
            
        except Exception as e:
            print(f"Warning: Batch tokenization failed: {e}")
            return {
                'input_ids': [],
                'length': [],
                'attention_mask': []
            }
    
    def tokenize_dataset_hf(self, 
                           dataset_path: str,
                           output_path: str,
                           batch_size: int = 5000,
                           min_length: int = 10,
                           max_length_filter: int = 10000) -> bool:
        """
        Tokenize HuggingFace dataset efficiently
        
        Args:
            dataset_path: Path to HuggingFace dataset directory
            output_path: Path to save tokenized data
            batch_size: Batch size for tokenization
            min_length: Minimum character length
            max_length_filter: Maximum character length for filtering
        """
        print(f"🚀 Tokenizing HuggingFace dataset: {dataset_path}")
        start_time = time.time()
        
        try:
            # Load dataset
            dataset, text_column = self.load_hf_dataset(dataset_path)
            
            # Tokenize using HuggingFace's efficient map function
            print(f"  ⚡ Tokenizing with batch size {batch_size}...")
            tokenize_start = time.time()
            
            def tokenize_function(batch):
                return self.tokenize_batch_hf(batch, text_column)
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                num_proc=self.num_workers,
                remove_columns=dataset.column_names,  # Remove original columns
                desc="Tokenizing"
            )
            
            tokenize_time = time.time() - tokenize_start
            
            print(f"  ✅ Successfully tokenized {len(tokenized_dataset):,} samples")
            
            if len(tokenized_dataset) == 0:
                print(f"  ❌ No valid tokenized samples")
                return False
            
            # Convert to Arrow table and save as Parquet
            print(f"  💾 Saving to {output_path}...")
            save_start = time.time()
            
            # Get the Arrow table from the dataset
            arrow_table = tokenized_dataset.data.table
            
            # Save as Parquet for efficiency
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            pq.write_table(arrow_table, f"{output_path}.parquet", compression='snappy')
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            # Performance stats
            valid_samples = len(tokenized_dataset)
            tokenize_rate = valid_samples / tokenize_time
            
            print(f"  ✅ Tokenization complete!")
            print(f"     Original samples: {len(dataset):,}")
            print(f"     Tokenized samples: {valid_samples:,}")
            print(f"     Tokenization time: {tokenize_time:.2f}s")
            print(f"     Save time: {save_time:.2f}s")
            print(f"     Total time: {total_time:.2f}s")
            print(f"     Rate: {tokenize_rate:.0f} samples/second")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Tokenization failed: {e}")
            return False
    
    def analyze_hf_datasets(self, dataset_paths: Dict[str, str]) -> Dict[str, Dict]:
        """Analyze HuggingFace dataset directories"""
        print("🔍 Analyzing HuggingFace datasets...")
        
        results = {}
        total_samples = 0
        
        for language, path in dataset_paths.items():
            print(f"\n📁 {language.upper()}: {path}")
            
            try:
                # Check if directory exists
                if not Path(path).exists():
                    results[language] = {'error': 'Directory does not exist', 'readable': False}
                    continue
                
                # Check for HuggingFace dataset files
                dataset_info = Path(path) / "dataset_info.json"
                state_file = Path(path) / "state.json"
                arrow_files = list(Path(path).glob("*.arrow"))
                
                if not (dataset_info.exists() or state_file.exists() or arrow_files):
                    results[language] = {'error': 'Not a HuggingFace dataset directory', 'readable': False}
                    continue
                
                # Try to load and analyze
                dataset = load_from_disk(path)
                
                # Find text column
                text_columns = [col for col in dataset.column_names 
                              if any(keyword in col.lower() for keyword in ['text', 'content', 'body'])]
                
                text_column = text_columns[0] if text_columns else dataset.column_names[0]
                
                # Sample text lengths
                sample_size = min(1000, len(dataset))
                sample_texts = dataset.select(range(sample_size))[text_column]
                text_lengths = [len(str(text)) for text in sample_texts if text]
                
                info = {
                    'readable': True,
                    'samples': len(dataset),
                    'columns': dataset.column_names,
                    'text_column': text_column,
                    'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                    'min_text_length': min(text_lengths) if text_lengths else 0,
                    'max_text_length': max(text_lengths) if text_lengths else 0
                }
                
                results[language] = info
                total_samples += len(dataset)
                
                print(f"  ✅ Samples: {len(dataset):,}")
                print(f"  ✅ Text column: '{text_column}'")
                print(f"  ✅ Avg text length: {info['avg_text_length']:.0f} chars")
                
            except Exception as e:
                results[language] = {'error': str(e), 'readable': False}
                print(f"  ❌ Error: {e}")
        
        print(f"\n📊 Total samples across all datasets: {total_samples:,}")
        return results

def process_all_hf_datasets(dataset_paths: Dict[str, str],
                           tokenizer_path: str,
                           output_base_path: str,
                           max_length: int = 2048,
                           batch_size: int = 5000,
                           num_workers: int = None) -> Dict[str, str]:
    """
    Process all HuggingFace datasets
    
    Returns:
        Dict mapping language -> output file path
    """
    # Initialize pipeline
    pipeline = HuggingFaceTokenizationPipeline(
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        num_workers=num_workers
    )
    
    # Analyze datasets first
    analysis = pipeline.analyze_hf_datasets(dataset_paths)
    
    # Filter readable datasets
    readable_datasets = {lang: path for lang, path in dataset_paths.items() 
                        if analysis[lang].get('readable', False)}
    
    if not readable_datasets:
        print("❌ No readable HuggingFace datasets found!")
        return {}
    
    print(f"\n🚀 Processing {len(readable_datasets)} HuggingFace datasets...")
    
    # Process each dataset
    output_paths = {}
    total_start = time.time()
    
    for language, dataset_path in readable_datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {language.upper()}")
        print(f"{'='*80}")
        
        output_path = f"{output_base_path}/{language}"
        
        success = pipeline.tokenize_dataset_hf(
            dataset_path=dataset_path,
            output_path=output_path,
            batch_size=batch_size,
            min_length=10,
            max_length_filter=10000  # Filter very long texts before tokenization
        )
        
        if success:
            output_paths[language] = f"{output_path}.parquet"
        else:
            print(f"❌ Failed to process {language}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"🎉 ALL DATASETS PROCESSED!")
    print(f"{'='*80}")
    print(f"Processed: {len(output_paths)}/{len(dataset_paths)} datasets")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if output_paths:
        print(f"\n📁 Tokenized datasets ready:")
        for lang, path in output_paths.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024*1024)
                # Count samples
                try:
                    table = pq.read_table(path, columns=['length'])
                    sample_count = len(table)
                    print(f"  ✅ {lang}: {sample_count:,} samples ({size_mb:.1f} MB) -> {path}")
                except:
                    print(f"  ✅ {lang}: {size_mb:.1f} MB -> {path}")
    
    return output_paths

def main():
    """Main function to process your HuggingFace datasets"""
    
    # Configuration for your specific setup
    config = {
        'tokenizer_path': "D:/fineweb-chunked/clean-bpe-tokenizer/tokenizer.json",
        
        # Your HuggingFace dataset directories
        'dataset_paths': {
            'combined': r'D:\fineweb-chunked\combined',
        },
        
        'output_base_path': "D:/fineweb-chunked/tokenized",
        'max_length': 2048,
        'batch_size': 5000,  # Large batches for efficiency
        'num_workers': mp.cpu_count()
    }
    
    # Verify tokenizer exists
    if not Path(config['tokenizer_path']).exists():
        print(f"❌ Tokenizer not found: {config['tokenizer_path']}")
        print("Please check the tokenizer path.")
        return
    
    # Process all datasets
    print("🧠 HuggingFace Dataset Tokenization Pipeline")
    print("="*80)
    
    output_paths = process_all_hf_datasets(
        dataset_paths=config['dataset_paths'],
        tokenizer_path=config['tokenizer_path'],
        output_base_path=config['output_base_path'],
        max_length=config['max_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    if output_paths:
        print(f"\n🎉 SUCCESS! Ready for training.")
        print(f"\nUse these tokenized datasets in your training:")
        print("tokenized_paths = {")
        for lang, path in output_paths.items():
            print(f"    '{lang}': r'{path}',")
        print("}")
        
        # Estimate total samples
        total_samples = 0
        for path in output_paths.values():
            try:
                table = pq.read_table(path, columns=['length'])
                total_samples += len(table)
            except:
                pass
        
        if total_samples > 0:
            print(f"\nTotal tokenized samples: {total_samples:,}")
            
            # Training time estimates
            batch_size = 32
            batches_per_epoch = total_samples // batch_size
            print(f"Batches per epoch (batch_size={batch_size}): {batches_per_epoch:,}")
    else:
        print(f"\n❌ No datasets were successfully processed.")

if __name__ == "__main__":
    main()
