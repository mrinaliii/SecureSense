import numpy as np
from datasets import load_metric
from transformers import Trainer

metric = load_metric("seqeval")


def compute_metrics(p):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [p for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    true_labels = [
        [l for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return metric.compute(
        predictions=true_predictions,
        references=true_labels
    )
