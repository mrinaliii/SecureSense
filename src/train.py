import torch
import pandas as pd
import os

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from data_loader import load_wikiann
from preprocess import tokenize_and_align_labels, tokenizer
import metrics
from metrics import compute_metrics


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = load_wikiann()

    print("\nTokenizing dataset...")

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags", "langs", "spans"]
    )

    label_list = dataset["train"].features["ner_tags"].feature.names
    metrics.label_list = label_list

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(

        output_dir="./results",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=4,
        num_train_epochs=3,

        learning_rate=2e-5,
        weight_decay=0.01,

        lr_scheduler_type="linear",
        warmup_ratio=0.1,

        fp16=True,
        dataloader_num_workers=0,

        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,
        logging_dir="./logs",

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...\n")

    trainer.train()

    os.makedirs("../models/distilbert-pii", exist_ok=True)
    trainer.save_model("../models/distilbert-pii")
    tokenizer.save_pretrained("../models/distilbert-pii")
    print("\nTraining Complete.")

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv("./results/training_metrics.csv", index=False)

    print("Metrics saved.")

    eval_df = df[df["eval_f1"].notna()]

    if not eval_df.empty:

        best = eval_df.sort_values("eval_f1", ascending=False).iloc[0]
        best.to_csv("results/best_metrics.csv")

        print("\nBEST MODEL:")
        print(best[["epoch","eval_loss","eval_precision","eval_recall","eval_f1"]])


if __name__ == "__main__":
    main()
