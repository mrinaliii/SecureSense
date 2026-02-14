from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="../models/distilbert-pii",
    aggregation_strategy="simple"
)

text = "John works at Microsoft in Seattle."

print(ner(text))
