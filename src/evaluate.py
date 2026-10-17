"""Shared evaluation helpers: metrics, reports, confusion matrices, plots.

Run `python -m src.evaluate` after training to build the model comparison
table (results/model_comparison.csv) from the saved metric files.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)

from .config import CATEGORIES, PLOTS_DIR, RESULTS_DIR  # noqa: E402


def compute_metrics(y_true, y_pred, labels=CATEGORIES) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="weighted", zero_division=0)
    return {"n_samples": int(len(y_true)), "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(p), 4), "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f), 4), "weighted_f1": round(float(fw), 4)}


def report_frame(y_true, y_pred, labels=CATEGORIES) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(4)


def plot_confusion(y_true, y_pred, title: str, fname: str, labels=CATEGORIES):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (folder label)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    return cm


def plot_loss_curves(log_history: list[dict], title: str, fname: str):
    tr = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in log_history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    if tr:
        ax.plot(*zip(*tr), marker=".", label="training loss")
    if ev:
        ax.plot(*zip(*ev), marker="o", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def build_comparison() -> pd.DataFrame:
    """Baseline vs fine-tuned transformer on the SAME test resumes.

    Reads results/predictions_baseline.csv and results/predictions_transformer.csv
    (written by train_baseline.py and train_transformer.py) and writes
    results/model_comparison.csv, results/model_comparison.md and
    results/plots/model_comparison.png.  Rows are also given for the supplied
    (real) and the synthetic test resumes separately."""
    files = {"TF-IDF + Logistic Regression": RESULTS_DIR / "predictions_baseline.csv",
             "Fine-tuned transformer": RESULTS_DIR / "predictions_transformer.csv"}
    rows, test_ids = [], {}
    for model, path in files.items():
        if not path.exists():
            print(f"[skip] {path.name} not found - run the training script first")
            continue
        pred = pd.read_csv(path)
        for split in ("test", "val"):
            sub = pred[pred["split"] == split]
            if split == "test":
                test_ids[model] = set(sub["resume_id"])
            groups = [("all", sub)]
            if "source_type" in sub.columns and sub["source_type"].notna().any():
                groups += [("supplied_original", sub[sub["source_type"] == "supplied_original"]),
                           ("synthetic", sub[sub["source_type"] == "synthetic"])]
            for name, g in groups:
                if len(g):
                    # subsets may not contain all 5 classes: average only over the classes present
                    labels = sorted(set(g["true"]) | set(g["predicted"]), key=CATEGORIES.index)
                    rows.append({"model": model, "split": split, "resumes": name,
                                 **compute_metrics(g["true"], g["predicted"], labels=labels)})
    df = pd.DataFrame(rows)
    if len(test_ids) == 2:
        a, b = test_ids.values()
        assert a == b, "baseline and transformer were evaluated on different test resumes"
    if len(df):
        df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        main = df[(df["split"] == "test") & (df["resumes"] == "all")]
        lines = ["# Baseline vs fine-tuned transformer", "",
                 f"Same resume-level split for both models; test set = {int(main['n_samples'].iloc[0])} resumes "
                 "(never used for training or model selection).", "",
                 "| model | split | resumes | n | accuracy | macro P | macro R | macro F1 | weighted F1 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in df.itertuples():
            lines.append(f"| {r.model} | {r.split} | {r.resumes} | {r.n_samples} | {r.accuracy:.3f} | "
                         f"{r.macro_precision:.3f} | {r.macro_recall:.3f} | {r.macro_f1:.3f} | {r.weighted_f1:.3f} |")
        lines += ["", "Small-data warning: one test resume changes accuracy by 5 percentage points; most resumes are "
                      "synthetic, so scores on synthetic resumes are optimistic for real-world resumes."]
        (RESULTS_DIR / "model_comparison.md").write_text("\n".join(lines))
        piv = main.set_index("model")[["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
        ax = piv.T.plot(kind="bar", figsize=(7.5, 3.8), rot=0, color=["#6C8EA4", "#0E7C86"][:len(piv)])
        ax.set_ylim(0, 1.05)
        ax.set_title("Test resumes: baseline vs fine-tuned transformer", fontsize=10)
        ax.grid(axis="y", alpha=.3)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
        plt.close()
    return df


if __name__ == "__main__":
    out = build_comparison()
    print(out.to_string(index=False))
    print("saved results/model_comparison.csv and results/model_comparison.md")
