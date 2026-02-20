from predict import TransformerPredictor

predictor = TransformerPredictor()

text = "John works at Microsoft in Seattle."

results = predictor.predict(text)

for r in results:
    print(r)