from model.transformer import SpikingMoELLM, SpikingLLM
import torch

# --- 3b
# num_params: 4,876,246,840
# moe_params: 3,088,497,720    
# regular_params: 1,459,282,944
# GPU Memory Usage:
#   Allocated: 18.61/30 GB        
#   Cached: 18.62/30 GB
#   Max Allocated: 18.61/30 GB   

# --- 1b
# num_params: 2,747,409,200
# moe_params: 1,686,859,824  
# regular_params: 797,750,784
# GPU Memory Usage:
#   Allocated: 10.39/30 GB      
#   Cached: 10/30.39 GB
#   Max Allocated: 10.39/30 GB 

def get_max_vram():
    max_memory_size = torch.cuda.get_device_properties("cuda").total_memory
    max_vram_gb = max_memory_size / (1024**3)
    return round(max_vram_gb, 2)

def print_gpu_memory_usage():
    """Prints the current GPU memory usage in PyTorch."""
    if torch.cuda.is_available():
        # Get the current GPU memory usage
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()
        max_memory = torch.cuda.max_memory_allocated()

        # Convert bytes to GB
        allocated_gb = allocated_memory / (1024 ** 3)
        cached_gb = cached_memory / (1024 ** 3)
        max_gb = max_memory / (1024 ** 3)
        vram = get_max_vram()

        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated_gb:.2f}/{vram} GB")
        print(f"  Cached: {cached_gb:.2f}/{vram} GB")
        print(f"  Max Allocated: {max_gb:.2f}/{vram} GB")
    else:
        print("CUDA is not available.  No GPU detected.")

if __name__ == '__main__':

    base_configs = {
        "1b": {
            "vocab_size": 128256,
            "d_model": 2048,
            "n_heads": 32,
            "n_kv_heads": 8,
            "num_layers": 24,
            "intermediate_size": 5504,
            "max_seq_len": 8192,
        },
        "3b": {
            "vocab_size": 128256,
            "d_model": 2560,
            "n_heads": 32,
            "n_kv_heads": 8,  
            "num_layers": 28,
            "intermediate_size": 6912,
            "max_seq_len": 16384,
        },
        "A": {
            "vocab_size": 0,
            "n_heads": 16,
            "num_layers": 28,
        }
    }
    
    llm = SpikingMoELLM(**base_configs["1b"]).to("cuda")
    num_params = llm.get_num_params()
    moe_params, regular_params = llm.count_moe_params()
    print(" --- 1B ")
    print(f"num_params: {num_params:,}")
    print(f"moe_params: {moe_params:,}")
    print(f"regular_params: {regular_params:,}")
    print_gpu_memory_usage()

    del llm 
    torch.cuda.empty_cache()

    llm = SpikingMoELLM(**base_configs["3b"]).to("cuda")
    num_params = llm.get_num_params()
    moe_params, regular_params = llm.count_moe_params()
    print(" --- 3B ")
    print(f"num_params: {num_params:,}")
    print(f"moe_params: {moe_params:,}")
    print(f"regular_params: {regular_params:,}")
    print_gpu_memory_usage()
    
    del llm 
    torch.cuda.empty_cache()

    llm = SpikingLLM(**base_configs["1b"]).to("cuda")
    num_params = llm.get_num_params()
    print(" --- 1B ")
    print(f"num_params: {num_params:,}")
    print_gpu_memory_usage()
    
    del llm 
    torch.cuda.empty_cache()

    llm = SpikingLLM(**base_configs["3b"]).to("cuda")
    num_params = llm.get_num_params()
    print(" --- 3B ")
    print(f"num_params: {num_params:,}")
    print_gpu_memory_usage()