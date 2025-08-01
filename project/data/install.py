from huggingface_hub import snapshot_download
folder = snapshot_download(
    "HuggingFaceFW/fineweb-2", 
    repo_type="dataset",
    local_dir="D:/fineweb2/",
    allow_patterns=[
        "data/fra_Latn/train/000_0000*", # french
        "data/fra_Latn/train/000_0001*", 
        "data/deu_Latn/train/000_0000*", # german
        "data/deu_Latn/train/000_0001*",
        "data/cmn_Hani/train/000_0000*", # chinese
        "data/cmn_Hani/train/000_0001*",
    ],
)

from huggingface_hub import snapshot_download
folder = snapshot_download(
    "HuggingFaceFW/fineweb", 
    repo_type="dataset",
    local_dir="D:/fineweb/",
    allow_patterns=[
        "data/CC-MAIN-2025-26/000_0000*", # english
        "data/CC-MAIN-2025-26/000_0001*",
    ],
)
