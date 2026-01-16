from datasets import load_dataset

ds = load_dataset("piyushsinghpasi/audiocaps-multilingual")
print(ds["test"].features)
