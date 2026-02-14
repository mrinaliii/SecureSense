import numpy as np
from evaluate import load

seqeval = load("seqeval")

label_list = None


def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):

        curr_preds = []
        curr_labels = []

        for p_i, l_i in zip(pred, lab):

            if l_i != -100:
                curr_preds.append(label_list[p_i])
                curr_labels.append(label_list[l_i])

        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
