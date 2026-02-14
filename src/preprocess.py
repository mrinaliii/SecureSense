from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    all_labels = examples["ner_tags"]

    new_labels = []

    for i, labels in enumerate(all_labels):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])

            else:
                label_ids.append(labels[word_idx])

            previous_word_idx = word_idx

        new_labels.append(label_ids)

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs
