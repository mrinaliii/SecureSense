import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Path Setup (Safe + Clean)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

csv_path = RESULTS_DIR / "training_metrics.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found.")

# ----------------------------
# Load Metrics
# ----------------------------
df = pd.read_csv(csv_path)

eval_df = df[df["eval_f1"].notna()].copy()
eval_df = eval_df.sort_values("epoch")

print("\nEvaluation Metrics Per Epoch:\n")
print(eval_df[[
    "epoch",
    "eval_loss",
    "eval_precision",
    "eval_recall",
    "eval_f1"
]])

# ----------------------------
# 1️⃣ F1 vs Epoch
# ----------------------------
plt.figure()
plt.plot(eval_df["epoch"], eval_df["eval_f1"])
plt.title("F1 vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "f1_vs_epoch.png")
plt.close()

# ----------------------------
# 2️⃣ Loss vs Epoch
# ----------------------------
plt.figure()
plt.plot(eval_df["epoch"], eval_df["eval_loss"])
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Eval Loss")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_vs_epoch.png")
plt.close()

# ----------------------------
# 3️⃣ Precision & Recall vs Epoch
# ----------------------------
plt.figure()
plt.plot(eval_df["epoch"], eval_df["eval_precision"])
plt.plot(eval_df["epoch"], eval_df["eval_recall"])
plt.title("Precision & Recall vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend(["Precision", "Recall"])
plt.tight_layout()
plt.savefig(RESULTS_DIR / "precision_recall_vs_epoch.png")
plt.close()

print("\nPlots successfully saved to top-level 'results/' folder.")