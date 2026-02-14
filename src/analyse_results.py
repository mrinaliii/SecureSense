import pandas as pd

df = pd.read_csv("results/training_metrics.csv")

eval_df = df[df["eval_f1"].notna()]

print(eval_df[[
    "epoch",
    "eval_loss",
    "eval_precision",
    "eval_recall",
    "eval_f1"
]])
import matplotlib.pyplot as plt

plt.plot(eval_df["epoch"], eval_df["eval_f1"])
plt.title("F1 vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.show()