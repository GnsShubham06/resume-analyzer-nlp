# Training README – dataset, fine-tuning, evaluation

All commands are run from the project folder (`resume-analyzer-nlp/`).

## 1. Dataset (100 resumes, 20 per category)

| Folder | Content |
|---|---|
| `data/raw/<Category>/` | exactly 20 resumes per category (Engineering, HR, Finance, Healthcare, Agriculture) |
| `data/resume_sources.csv` | one row per document: filename, category, `source_type` (`supplied_original` / `synthetic`), provenance, licence, status |
| `data/excluded/` | 7 supplied resumes whose text does not match their folder label (moved, **not** relabelled) + `excluded_resumes.csv` with the reason |

* 18 resumes are originals supplied by the team (their public source and licence were not documented – see `resume_sources.csv`).
* 82 resumes are **synthetic**: fictional people, employers and institutions, generated reproducibly by
  `python scripts/prepare_dataset_100.py` (seed 2026) from hand-written content pools in `scripts/synthetic_content.py`.
  They are marked as synthetic in the file name (`SYN_*.pdf`), in the PDF metadata, in a visible footer line
  (removed again during preprocessing so a model cannot learn it) and in `resume_sources.csv`.
* 100 resumes is enough for a student prototype but still a very small dataset for a 125-million-parameter model.

Check the dataset before training:

```bash
python -m src.validate_dataset      # counts, empty files, duplicates, near-duplicates, provenance, label check
python -m src.build_dataset         # extraction, cleaning, anonymisation -> data/resume_dataset.csv
```

## 2. Split (same for both models)

`src/data_split.py → make_splits()` splits by **resume**, stratified by category and source type:
12 train / 4 validation / 4 test resumes per category = **60 / 20 / 20**. The split is saved in
`data/processed/splits.csv` and reused by the baseline and the transformer.

## 3. Baseline

```bash
python -m src.train_baseline        # TF-IDF + Logistic Regression, trained on the 60 training resumes
```

## 4. Fine-tuning the transformer

```bash
python -m src.train_transformer --smoke   # 1 epoch, 2 batches: checks that everything runs (~1 min)
python -m src.train_transformer           # full run (defaults below)
```

Defaults: `roberta-base`, 6 epochs, learning rate 5e-5, batch size 2, max 512 tokens per resume, AdamW
(weight decay 0.01), linear schedule with 10 % warm-up, all parameters trainable, seed 42.
Options: `--freeze_embeddings`, `--freeze_layers N`, `--epochs`, `--lr`, `--batch_size`, `--max_length`.

How the defaults were chosen (validation set only, test set not looked at): run 1 used 4 epochs at 2e-5 and
was still improving at the last epoch (validation macro-F1 0.50); run 2 used 6 epochs at 5e-5 and reached
validation macro-F1 1.00. Both runs' logs are kept in `results/tuning/`. Run 2 is the reported model.
On our CPU container one epoch took about 3-3.5 minutes (about 20 minutes for 6 epochs plus evaluation); batch size 4 ran out of memory on an 8 GB machine.

The model name `roberta-base` is downloaded from the Hugging Face Hub. If the Hub is unreachable, the script
falls back to `models/pretrained/roberta-base-spacy` (create it once with
`pip install spacy-curated-transformers && python -m spacy download en_core_web_trf && python scripts/convert_spacy_roberta.py`).
The model actually used is written to `results/transformer_verification.json`.

The training loop is written out in `src/train_transformer.py`:
`train_one_epoch()` → forward pass → `loss_fn(logits, labels)` → `loss.backward()` → `optimizer.step()`.

### Outputs

| File | What it shows |
|---|---|
| `models/transformer_finetuned/` | best checkpoint (by validation macro-F1): `model.safetensors`, `config.json`, tokenizer files, `label_map.json` |
| `results/transformer_training_log.csv` | per epoch: training loss, validation loss, validation accuracy and macro-F1, saved or not |
| `results/transformer_step_log.csv` | loss and learning rate of every optimiser step |
| `results/transformer_verification.json` | pretrained weights loaded, trainable parameters, weight change after the first `optimizer.step()`, number of changed tensors, reload check |
| `results/predictions_transformer.csv` | validation and test predictions of the reloaded checkpoint |
| `results/plots/transformer_loss_curve.png` | loss curves |

## 5. Comparison

```bash
python -m src.evaluate              # results/model_comparison.csv + model_comparison.md + plots/model_comparison.png
```

Both models are scored on the same 20 test resumes (accuracy, macro precision/recall/F1, weighted F1,
confusion matrices), also separately for the supplied and the synthetic test resumes.
BERTScore is not used: it compares generated text with reference text and does not apply to classification.

## 6. Application

```bash
streamlit run app.py                # loads models/transformer_finetuned/ and the baseline
python -m src.predict data/raw/Engineering/10985403.pdf
```
