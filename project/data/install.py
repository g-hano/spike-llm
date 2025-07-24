from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb-2", 
                repo_type="dataset",
                local_dir="./fineweb2/",
                allow_patterns=["data/tur_Latn/train/000_00000*"])
