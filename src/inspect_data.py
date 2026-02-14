from data_loader import load_wikiann

dataset = load_wikiann()

print(dataset)
print(dataset["train"][0])
