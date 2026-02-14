from regex_detector import regex_detect
from transformers import pipeline


ner_pipeline = pipeline(
    "token-classification",
    model="../models/distilbert-pii",
    aggregation_strategy="simple"
)


def hybrid_detect(text):

    regex_results = regex_detect(text)
    bert_results = ner_pipeline(text)

    return {
        "regex_detection": regex_results,
        "bert_detection": bert_results
    }
