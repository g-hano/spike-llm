from datasets import load_dataset
'''
english: 19.094.000 rows
chinese: 39.237.000 rows
french: 66.626.000 rows
german: 65.980.000 rows
'''

dataset = load_dataset(r"D:\fineweb2\data/en")
dataset = dataset.remove_columns(['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'])
print(dataset)
dataset.save_to_disk(r"D:\fineweb2-train/en")

for lang in [
        'fra_Latn', 
        'deu_Latn', 
        'cmn_Hani']:
    dataset = load_dataset(r"D:\fineweb2\data/" + lang)
    print(dataset)
    dataset = dataset.remove_columns(['id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'language_script', 'minhash_cluster_size', 'top_langs'])
    print(dataset)

    dataset.save_to_disk(r"D:\fineweb2-train/" + lang)
