from datasets import load_dataset


def load_wikiann():
    dataset = load_dataset("wikiann", "en")
    return dataset
