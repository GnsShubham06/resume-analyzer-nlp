"""Practical 10 - FINE-TUNING a pretrained transformer for 5-class resume classification.

This file is the complete fine-tuning implementation.  It uses Hugging Face
Transformers to load the pretrained model and tokenizer, and a plain PyTorch
training loop so that every step is visible in the source code:

  STEP 1   load_pretrained_model()   pretrained RoBERTa-base + matching tokenizer
  STEP 2   load_split_data()         resume texts, category -> number (config.LABEL2ID)
  STEP 3   make_loader()             tokenisation with truncation and padding
  STEP 4   load_split_data()         resume-level train / validation / test split (data_split.py)
  STEP 5   load_pretrained_model()   sequence-classification model with 5 output labels
  STEP 6   set_trainable_parameters()  which parameters are updated
  STEP 7   main()                    AdamW optimiser, linear warm-up/decay schedule, CrossEntropyLoss
  STEP 8   train_one_epoch()         forward pass + classification loss
  STEP 9   train_one_epoch()         loss.backward()   (backpropagation)
  STEP 10  train_one_epoch()         optimizer.step()  (parameter update)
  STEP 11  main()                    validation after every epoch, best checkpoint saved
  STEP 12  verify_saved_checkpoint() a fresh model is loaded from disk and evaluated

Proof that fine-tuning really happens is written to
results/transformer_verification.json (pretrained weights loaded, trainable
parameter count, a weight that changes after optimizer.step(), a frozen
weight that does not, and a reload check).

Run (full):   python -m src.train_transformer
Run (smoke):  python -m src.train_transformer --smoke      (1 epoch, 2 batches, separate folder)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import (CATEGORIES, DEFAULT_TRANSFORMER, FINETUNED_MODEL_DIR, ID2LABEL, LABEL2ID,
                     LOCAL_ROBERTA_DIR, MODELS_DIR, PLOTS_DIR, RANDOM_SEED, RESULTS_DIR)
from .data_split import get_splits, load_dataset
from .evaluate import compute_metrics, plot_confusion, report_frame

TEXT_COLUMN = "transformer_text"     # cleaned, anonymised natural text (see preprocessing.py)


def set_seed(seed: int):
    """Same seed -> same split, same initial classification head, same batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------- STEP 1 + 5
def load_pretrained_model(model_name: str):
    """Load the pretrained tokenizer and the pretrained encoder with a new
    5-class classification head.  Returns (tokenizer, model, model_used, loading_info).

    If the Hugging Face Hub cannot be reached and 'roberta-base' was requested,
    the locally converted RoBERTa-base (scripts/convert_spacy_roberta.py) is used
    instead; this fallback is recorded in the verification report."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    used, fallback = model_name, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as err:
        if model_name == "roberta-base" and (Path(LOCAL_ROBERTA_DIR) / "config.json").exists():
            used = str(LOCAL_ROBERTA_DIR)
            fallback = (f"'roberta-base' could not be downloaded ({type(err).__name__}: Hugging Face Hub "
                        f"unreachable); used the local RoBERTa-base converted from spaCy en_core_web_trf: {used}")
            print("[fallback]", fallback)
            tokenizer = AutoTokenizer.from_pretrained(used)
        else:
            raise SystemExit(f"Cannot load '{model_name}': {err}\nRemedy: connect to the internet (Colab works), "
                             f"or run `python scripts/convert_spacy_roberta.py` and pass "
                             f"--model_name {LOCAL_ROBERTA_DIR}")
    model, info = AutoModelForSequenceClassification.from_pretrained(
        used, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID, output_loading_info=True)
    loading_info = {k: sorted(v) if isinstance(v, (set, list)) else str(v) for k, v in info.items()}
    return tokenizer, model, used, fallback, loading_info


# ------------------------------------------------------------------ STEP 6
def set_trainable_parameters(model, freeze_embeddings: bool = False, freeze_layers: int = 0) -> dict:
    """Default: FULL fine-tuning - every pretrained weight and the new head are
    trainable.  Optionally freeze the embedding layer and/or the lowest
    `freeze_layers` encoder layers (requires_grad=False -> never updated)."""
    base = getattr(model, model.base_model_prefix)          # e.g. model.roberta
    for p in model.parameters():
        p.requires_grad = True
    if freeze_embeddings:
        for p in base.embeddings.parameters():
            p.requires_grad = False
    if freeze_layers > 0:
        layers = base.encoder.layer if hasattr(base, "encoder") else base.transformer.layer
        for layer in layers[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable,
            "trainable_percent": round(100 * trainable / total, 2),
            "frozen_embeddings": freeze_embeddings, "frozen_encoder_layers": freeze_layers}


# --------------------------------------------------------------- STEP 2 + 4
def load_split_data() -> dict:
    """Resume texts with numeric labels, split by RESUME (splits.csv) so the
    same resume can never be in two splits.  Test resumes are only used once,
    at the very end, with the saved best checkpoint."""
    df = load_dataset()
    split = get_splits(df).set_index("resume_id")["split"]
    df = df.assign(split=df["resume_id"].map(split), label=df["category"].map(LABEL2ID))
    assert df["label"].notna().all(), "unknown category in dataset"
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    overlap = set(parts["train"].resume_id) & (set(parts["val"].resume_id) | set(parts["test"].resume_id))
    assert not overlap, f"data leakage: {overlap}"
    return parts


# ------------------------------------------------------------------ STEP 3
def make_loader(frame: pd.DataFrame, tokenizer, max_length: int, batch_size: int, shuffle: bool, seed: int):
    """Tokenise every resume ONCE with truncation (first `max_length` tokens),
    then pad each batch to its longest resume inside collate()."""
    enc = tokenizer(frame[TEXT_COLUMN].fillna("").tolist(), truncation=True, max_length=max_length)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": int(frame["label"].iloc[i])} for i in range(len(frame))]

    def collate(batch):
        padded = tokenizer.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch],
                               padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([b["labels"] for b in batch])
        return padded

    gen = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=gen)


# -------------------------------------------------------------- STEP 8-10
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch, step_log,
                    watch=None, max_batches=None):
    model.train()                                   # dropout on
    losses = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()                       # clear gradients of the previous batch
        # STEP 8: forward pass -> one score (logit) per class, then the classification loss
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = loss_fn(logits, batch["labels"])     # cross-entropy between scores and true labels
        # STEP 9: backpropagation - gradient of the loss for every trainable weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        # proof of learning: copy the watched weights just before the first update with lr > 0
        check_now = watch is not None and not watch.get("done") and optimizer.param_groups[0]["lr"] > 0
        if check_now:
            snapshot_watched(model, watch, step=len(step_log) + 1, lr=optimizer.param_groups[0]["lr"])
        # STEP 10: the optimiser changes the weights in the direction that lowers the loss
        optimizer.step()
        scheduler.step()
        if check_now:
            compare_watched(model, watch)
        losses.append(loss.item())
        if (i + 1) % 5 == 0:
            print(f"  epoch {epoch} batch {i + 1}/{len(loader)} loss {loss.item():.4f}", flush=True)
        step_log.append({"epoch": epoch, "step": len(step_log) + 1, "train_loss": round(loss.item(), 5),
                         "learning_rate": scheduler.get_last_lr()[0]})
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Validation / test: no gradients, no weight updates."""
    model.eval()                                    # dropout off
    all_logits, losses = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        losses.append(loss_fn(logits, batch["labels"]).item() * len(batch["labels"]))
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=-1).numpy()
    pred = [ID2LABEL[int(i)] for i in probs.argmax(1)]
    metrics = compute_metrics(frame["category"].tolist(), pred)
    metrics["loss"] = round(sum(losses) / len(frame), 5)
    out = pd.DataFrame({"resume_id": frame["resume_id"], "source_type": frame.get("source_type"),
                        "true": frame["category"], "predicted": pred, "confidence": probs.max(1).round(4)})
    for j, c in enumerate(CATEGORIES):
        out[f"p_{c}"] = probs[:, j].round(4)
    return metrics, out


# --------------------------------------------- proof that weights change
def make_weight_watch(model) -> dict:
    """Choose the tensors to watch: one weight matrix of the NEW classification
    head, one of the TOP pretrained encoder layer (attention query) and, if any
    layer is frozen, one frozen tensor that must stay unchanged."""
    named = list(model.named_parameters())
    head = [n for n, p in named if p.requires_grad and "classifier" in n and n.endswith("weight")]
    enc = [n for n, p in named if p.requires_grad and "classifier" not in n
           and (n.endswith("query.weight") or n.endswith("q_lin.weight"))]
    frozen = [n for n, p in named if not p.requires_grad and n.endswith("weight")]
    pick = {}
    if head:
        pick["new classification head"] = head[-1]
    if enc:
        pick["pretrained top encoder layer"] = enc[-1]
    if frozen:
        pick["frozen parameter (must not change)"] = frozen[0]
    return {"names": pick}


def snapshot_watched(model, watch: dict, step: int, lr: float):
    """.detach().clone() makes an INDEPENDENT copy of the current values; a plain
    reference would change together with the parameter and prove nothing."""
    params = dict(model.named_parameters())
    watch["before"] = {k: params[n].detach().clone().cpu() for k, n in watch["names"].items()}
    watch["step"], watch["lr"] = step, lr


def compare_watched(model, watch: dict):
    params = dict(model.named_parameters())
    res = {"optimizer_step": watch["step"], "learning_rate_used": watch["lr"]}
    for k, n in watch["names"].items():
        after = params[n].detach().cpu()
        diff = (after - watch["before"][k]).abs()
        res[k] = {"parameter": n, "shape": list(after.shape), "max_abs_change": float(diff.max()),
                  "mean_abs_change": float(diff.mean()), "changed": bool(diff.max() > 0),
                  "requires_grad": params[n].requires_grad}
    watch["after_first_update"] = res
    watch["done"] = True
    del watch["before"]


def compare_with_pretrained(initial: dict, model) -> dict:
    changed, total, max_change = 0, 0, 0.0
    for n, p in model.named_parameters():
        if n not in initial:
            continue
        total += 1
        d = float((p.detach().cpu() - initial[n]).abs().max())
        changed += d > 0
        max_change = max(max_change, d)
    return {"pretrained_tensors_compared": total, "tensors_changed_by_training": int(changed),
            "largest_single_weight_change": max_change}


# ----------------------------------------------------------------- STEP 12
def verify_saved_checkpoint(out_dir: Path, tokenizer_len: int, val_loader, loss_fn, device, val_frame, expected: dict):
    """Load a FRESH model object from the saved folder and check that it gives
    exactly the validation result recorded when the checkpoint was saved."""
    from transformers import AutoModelForSequenceClassification
    fresh = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    m, _ = evaluate(fresh, val_loader, loss_fn, device, val_frame)
    same = abs(m["macro_f1"] - expected["macro_f1"]) < 1e-9 and abs(m["loss"] - expected["loss"]) < 1e-4
    return fresh, {"reloaded_from": str(out_dir), "val_macro_f1_reloaded": m["macro_f1"],
                   "val_loss_reloaded": m["loss"], "val_macro_f1_at_save": expected["macro_f1"],
                   "val_loss_at_save": expected["loss"], "identical": bool(same)}


def plot_curves(epoch_log: pd.DataFrame, step_log: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    if len(step_log):
        within = step_log.groupby("epoch").cumcount() + 1
        x = step_log["epoch"] - 1 + within / step_log.groupby("epoch")["step"].transform("count")
        ax.plot(x, step_log["train_loss"], color="#9bbfc7", lw=1, label="training loss (per batch)")
    ax.plot(epoch_log["epoch"], epoch_log["train_loss"], "o-", color="#0e7c86", label="training loss (epoch mean)")
    ax.plot(epoch_log["epoch"], epoch_log["val_loss"], "s-", color="#c8553d", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained transformer on the resume dataset")
    ap.add_argument("--model_name", default=DEFAULT_TRANSFORMER, help="Hugging Face model id or local folder")
    # defaults = the settings chosen on the VALIDATION resumes (see results/tuning/):
    # run 1 (lr 2e-5, 4 epochs) was still under-trained, run 2 (lr 5e-5, 6 epochs) was kept
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=2, help="small batches keep RAM below ~4 GB on a CPU laptop")
    ap.add_argument("--max_length", type=int, default=512, help="tokens kept per resume (truncation)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--freeze_embeddings", action="store_true")
    ap.add_argument("--freeze_layers", type=int, default=0, help="freeze the lowest N encoder layers")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--output_dir", default=str(FINETUNED_MODEL_DIR))
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches, writes to *_smoke folders")
    args = ap.parse_args()

    res_dir = RESULTS_DIR
    if args.smoke:
        args.epochs, args.output_dir = 1, str(MODELS_DIR / "transformer_smoke")
        res_dir = RESULTS_DIR / "smoke"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # STEP 1 + 5: pretrained tokenizer and model (5 output labels)
    tokenizer, model, model_used, fallback, loading_info = load_pretrained_model(args.model_name)
    newly_init = loading_info.get("missing_keys", [])
    print(f"loaded pretrained '{model_used}'. Newly initialised (task head only): {newly_init}")
    # STEP 6: trainable parameters
    param_info = set_trainable_parameters(model, args.freeze_embeddings, args.freeze_layers)
    print(f"parameters: total {param_info['total_parameters']:,}, trainable {param_info['trainable_parameters']:,} "
          f"({param_info['trainable_percent']}%)")
    model.to(device)
    initial_weights = {n: p.detach().clone().cpu() for n, p in model.named_parameters() if "classifier" not in n}

    # STEP 2-4: data, labels, split, tokenisation
    parts = load_split_data()
    print({s: f"{len(f)} resumes" for s, f in parts.items()})
    train_loader = make_loader(parts["train"], tokenizer, args.max_length, args.batch_size, True, args.seed)
    val_loader = make_loader(parts["val"], tokenizer, args.max_length, 4, False, args.seed)
    test_loader = make_loader(parts["test"], tokenizer, args.max_length, 4, False, args.seed)

    # STEP 7: loss, optimiser, learning-rate schedule
    from transformers import get_linear_schedule_with_warmup
    loss_fn = torch.nn.CrossEntropyLoss()
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm"))]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm"))]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    max_batches = 2 if args.smoke else None
    steps_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    # STEP 8-11: training loop with validation after every epoch
    watch = make_weight_watch(model)
    step_log, epoch_log = [], []
    best = {"macro_f1": -1.0, "loss": float("inf"), "epoch": None}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch, step_log,
                                     watch=watch, max_batches=max_batches)
        val_m, _ = evaluate(model, val_loader, loss_fn, device, parts["val"])
        improved = (val_m["macro_f1"] > best["macro_f1"]) or \
                   (val_m["macro_f1"] == best["macro_f1"] and val_m["loss"] < best["loss"])
        if improved:                                 # STEP 11: keep only the best epoch (by VALIDATION)
            best = {"macro_f1": val_m["macro_f1"], "loss": val_m["loss"], "epoch": epoch}
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        epoch_log.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": val_m["loss"],
                          "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
                          "learning_rate_end": scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 1),
                          "saved_as_best": improved})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_m['loss']:.4f} "
              f"val_acc={val_m['accuracy']:.3f} val_macroF1={val_m['macro_f1']:.3f} {'<- saved' if improved else ''}")
    train_seconds = round(time.time() - t_start, 1)
    after_training = compare_with_pretrained(initial_weights, model)

    meta = {"label2id": LABEL2ID, "id2label": ID2LABEL, "text_column": TEXT_COLUMN, "max_length": args.max_length,
            "base_model_requested": args.model_name, "base_model_used": model_used, "best_epoch": best["epoch"],
            "selection": "highest validation macro-F1 (tie -> lower validation loss)", "hyperparameters": vars(args)}
    (out_dir / "label_map.json").write_text(json.dumps(meta, indent=1))

    epoch_df, step_df = pd.DataFrame(epoch_log), pd.DataFrame(step_log)
    epoch_df.to_csv(res_dir / "transformer_training_log.csv", index=False)
    step_df.to_csv(res_dir / "transformer_step_log.csv", index=False)
    plot_curves(epoch_df, step_df, PLOTS_DIR / ("transformer_loss_curve_smoke.png" if args.smoke else
                                                 "transformer_loss_curve.png"),
                f"Fine-tuning {Path(model_used).name}: loss per epoch")

    # STEP 12: fresh model from disk -> reload check, then the ONLY use of the test set
    fresh, reload_check = verify_saved_checkpoint(out_dir, len(tokenizer), val_loader, loss_fn, device, parts["val"], best)
    test_m, test_pred = evaluate(fresh, test_loader, loss_fn, device, parts["test"])
    val_m, val_pred = evaluate(fresh, val_loader, loss_fn, device, parts["val"])
    pd.concat([test_pred.assign(split="test"), val_pred.assign(split="val")]).to_csv(
        res_dir / "predictions_transformer.csv", index=False)
    report_frame(test_pred["true"], test_pred["predicted"]).to_csv(res_dir / "classification_report_transformer_test.csv")
    if not args.smoke:
        plot_confusion(test_pred["true"], test_pred["predicted"],
                       f"Fine-tuned transformer - test resumes (n={len(test_pred)})", "confusion_transformer_test.png")

    verification = {
        "model_requested": args.model_name, "model_used": model_used, "fallback": fallback,
        "pretrained_loading": {"newly_initialised_tensors": newly_init,
                               "unexpected_keys": loading_info.get("unexpected_keys", []),
                               "meaning": "every tensor except the new classification head was loaded from the "
                                          "pretrained checkpoint"},
        "parameters": param_info,
        "weight_change_after_first_optimizer_step": watch.get("after_first_update"),
        "weight_change_after_training": after_training,
        "training": {"device": str(device), "epochs_run": args.epochs, "optimizer_steps": len(step_log),
                     "train_seconds": train_seconds, "best_epoch": best["epoch"],
                     "first_batch_loss": step_log[0]["train_loss"] if step_log else None,
                     "last_batch_loss": step_log[-1]["train_loss"] if step_log else None},
        "checkpoint": {"directory": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())},
        "reload_check": reload_check,
        "smoke_test": args.smoke,
    }
    (res_dir / "transformer_verification.json").write_text(json.dumps(verification, indent=1))
    metrics = {"model": f"Fine-tuned {Path(model_used).name}", "base_model_requested": args.model_name,
               "base_model_used": model_used, "hyperparameters": vars(args), "best_epoch": best["epoch"],
               "results": {"val": val_m, "test": test_m}, "n_resumes": {s: len(f) for s, f in parts.items()},
               "smoke_test": args.smoke}
    (res_dir / "metrics_transformer.json").write_text(json.dumps(metrics, indent=1))

    print("\nweight change after the first optimizer.step():")
    for k, v in (watch.get("after_first_update") or {}).items():
        if isinstance(v, dict):
            print(f"  {k:36s} {v['parameter']}: max |change| = {v['max_abs_change']:.3e}  changed={v['changed']}")
    print(f"tensors changed by training: {after_training['tensors_changed_by_training']} / "
          f"{after_training['pretrained_tensors_compared']}")
    print(f"best epoch (validation): {best['epoch']}; checkpoint: {out_dir}; reload identical: {reload_check['identical']}")
    print(f"TEST (used once, best checkpoint): accuracy={test_m['accuracy']:.3f} macroF1={test_m['macro_f1']:.3f}")
    print(f"logs: {res_dir/'transformer_training_log.csv'}, {res_dir/'transformer_verification.json'}")


if __name__ == "__main__":
    main()
