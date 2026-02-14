from data_loader import load_wikiann
from preprocess import tokenize_and_align_labels

dataset = load_wikiann()

tokenized = dataset["train"][0]
aligned = tokenize_and_align_labels(tokenized)

print(aligned.keys())
