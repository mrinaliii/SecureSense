from preprocess import tokenizer

tokenizer.save_pretrained("../models/distilbert-pii")

print("Tokenizer saved successfully.")