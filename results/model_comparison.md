# Baseline vs fine-tuned transformer

Same resume-level split for both models; test set = 20 resumes (never used for training or model selection).

| model | split | resumes | n | accuracy | macro P | macro R | macro F1 | weighted F1 |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | test | all | 20 | 0.900 | 0.920 | 0.900 | 0.898 | 0.898 |
| TF-IDF + Logistic Regression | test | supplied_original | 5 | 0.600 | 0.400 | 0.600 | 0.467 | 0.467 |
| TF-IDF + Logistic Regression | test | synthetic | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TF-IDF + Logistic Regression | val | all | 20 | 0.950 | 0.960 | 0.950 | 0.949 | 0.949 |
| TF-IDF + Logistic Regression | val | supplied_original | 4 | 0.750 | 0.625 | 0.750 | 0.667 | 0.667 |
| TF-IDF + Logistic Regression | val | synthetic | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | test | all | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | test | supplied_original | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | test | synthetic | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | val | all | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | val | supplied_original | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tuned transformer | val | synthetic | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Small-data warning: one test resume changes accuracy by 5 percentage points; most resumes are synthetic, so scores on synthetic resumes are optimistic for real-world resumes.