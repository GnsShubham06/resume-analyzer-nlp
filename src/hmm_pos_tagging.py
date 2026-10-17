"""Practical 8: Hidden Markov Model POS tagging.

* Training corpus: NLTK's Penn Treebank sample (3,914 tagged WSJ sentences,
  ~100k tokens) mapped to the 12-tag Universal tagset.  It is newswire, not
  resumes - there is no POS-annotated resume corpus, so this is a clearly
  stated data limitation.  90 % train / 10 % held-out test.
* Model (implemented from scratch): bigram HMM
      P(tags, words) = prod P(t_i | t_{i-1}) * P(w_i | t_i)
  with add-k smoothing; unknown words are handled by suffix/shape classes.
* Decoding: Viterbi algorithm (log-space dynamic programming).
* Reference: NLTK's HiddenMarkovModelTagger trained on the same data.
* Application: tag real sentences extracted from the resumes.

Run: python -m src.hmm_pos_tagging
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import DATASET_CSV, PLOTS_DIR, RESULTS_DIR
from .nlp_utils import ensure_nltk
from .preprocessing import clean_text, split_sentences

OUT = RESULTS_DIR / "practical8_hmm"


def unk_class(word: str) -> str:
    if re.fullmatch(r"[\d.,/:-]+", word):
        return "<NUM>"
    for suf, cls in (("ing", "<ING>"), ("ed", "<ED>"), ("ly", "<LY>"), ("tion", "<TION>"),
                     ("ment", "<MENT>"), ("s", "<S>"), ("er", "<ER>"), ("al", "<AL>"), ("ive", "<IVE>")):
        if word.lower().endswith(suf):
            return cls
    if word[:1].isupper():
        return "<CAP>"
    return "<UNK>"


class HMMTagger:
    def __init__(self, k: float = 0.01, min_count: int = 1):
        self.k, self.min_count = k, min_count

    def _norm(self, w):
        w2 = w.lower()
        return w2 if w2 in self.vocab else unk_class(w)

    def fit(self, tagged_sents):
        wc = Counter(w.lower() for s in tagged_sents for w, _ in s)
        self.vocab = {w for w, c in wc.items() if c > self.min_count}
        self.tags = sorted({t for s in tagged_sents for _, t in s})
        T = len(self.tags)
        self.ti = {t: i for i, t in enumerate(self.tags)}
        trans = np.zeros((T + 1, T))           # last row = start state
        emit = defaultdict(lambda: np.zeros(T))
        for s in tagged_sents:
            prev = T
            for w, t in s:
                j = self.ti[t]
                trans[prev, j] += 1
                emit[self._norm(w)][j] += 1
                prev = j
        self.trans_counts = trans
        self.log_trans = np.log((trans + self.k) / (trans + self.k).sum(axis=1, keepdims=True))
        tag_tot = sum(emit.values())
        V = len(emit)
        self.log_emit = {w: np.log((c + self.k) / (tag_tot + self.k * V)) for w, c in emit.items()}
        self.log_emit_unseen = np.log(self.k / (tag_tot + self.k * V))
        return self

    def viterbi(self, words):
        T, n = len(self.tags), len(words)
        if n == 0:
            return []
        V = np.full((n, T), -math.inf)
        B = np.zeros((n, T), dtype=int)
        e = lambda w: self.log_emit.get(self._norm(w), self.log_emit_unseen)  # noqa: E731
        V[0] = self.log_trans[T] + e(words[0])
        for i in range(1, n):
            scores = V[i - 1][:, None] + self.log_trans[:T]      # (prev, cur)
            B[i] = scores.argmax(axis=0)
            V[i] = scores.max(axis=0) + e(words[i])
        path = [int(V[-1].argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(B[i, path[-1]]))
        return [self.tags[j] for j in reversed(path)]

    def tag(self, words):
        return list(zip(words, self.viterbi(words)))

    def accuracy(self, tagged_sents):
        ok = tot = 0
        for s in tagged_sents:
            pred = self.viterbi([w for w, _ in s])
            ok += sum(p == t for p, (_, t) in zip(pred, s))
            tot += len(s)
        return ok / tot


def main():
    ensure_nltk()
    import nltk
    from nltk.corpus import treebank
    OUT.mkdir(parents=True, exist_ok=True)
    sents = [[(w, t) for w, t in s if t != "X"] for s in treebank.tagged_sents(tagset="universal")]
    sents = [s for s in sents if s]
    cut = int(0.9 * len(sents))
    train, test = sents[:cut], sents[cut:]

    hmm = HMMTagger().fit(train)
    acc_own = hmm.accuracy(test)

    nltk_hmm = nltk.tag.hmm.HiddenMarkovModelTrainer().train_supervised(
        train, estimator=lambda fd, bins: nltk.probability.LidstoneProbDist(fd, 0.1, bins))
    acc_nltk = nltk_hmm.accuracy(test) if hasattr(nltk_hmm, "accuracy") else nltk_hmm.evaluate(test)

    # baseline: most-frequent tag per word
    mft = Counter()
    wt = defaultdict(Counter)
    for s in train:
        for w, t in s:
            wt[w.lower()][t] += 1
            mft[t] += 1
    default = mft.most_common(1)[0][0]
    ok = tot = 0
    for s in test:
        for w, t in s:
            ok += (wt[w.lower()].most_common(1)[0][0] if w.lower() in wt else default) == t
            tot += 1
    acc_mft = ok / tot

    df = pd.read_csv(DATASET_CSV)
    resume_tags = []
    for rid in ["HR_03", "ENG_05", "FIN_02", "HLT_02", "AGR_02"]:
        text = clean_text(df.set_index("resume_id").loc[rid, "anonymized_text"])
        s = next(x for x in split_sentences(text) if 6 <= len(x.split()) <= 15 and "[" not in x)
        words = nltk.word_tokenize(s)
        ours = hmm.tag(words)
        ref = nltk.pos_tag(words, tagset="universal")   # perceptron tagger as a reference
        agree = sum(a[1] == b[1] for a, b in zip(ours, ref)) / len(words)
        resume_tags.append({"resume_id": rid, "sentence": s, "hmm_tags": ours,
                            "perceptron_tags": ref, "agreement_with_perceptron": round(agree, 3)})
        print(f"\n[{rid}] {s}\n HMM : " + " ".join(f"{w}/{t}" for w, t in ours) +
              "\n Perc: " + " ".join(f"{w}/{t}" for w, t in ref) + f"\n agreement={agree:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    P = np.exp(hmm.log_trans)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(P, xticklabels=hmm.tags, yticklabels=hmm.tags + ["<START>"], cmap="mako", annot=True,
                fmt=".2f", annot_kws={"size": 6}, ax=ax)
    ax.set_xlabel("next tag")
    ax.set_ylabel("previous tag")
    ax.set_title("HMM transition probabilities P(t_i | t_{i-1})", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hmm_transition_matrix.png", dpi=150)
    plt.close(fig)

    report = {"corpus": "NLTK Penn Treebank sample, universal tagset", "train_sentences": len(train),
              "test_sentences": len(test), "train_tokens": sum(map(len, train)),
              "test_tokens": sum(map(len, test)), "tagset": hmm.tags,
              "accuracy": {"own_hmm_viterbi": round(acc_own, 4), "nltk_hmm": round(float(acc_nltk), 4),
                           "most_frequent_tag_baseline": round(acc_mft, 4)},
              "resume_sentences": resume_tags,
              "limitation": "trained on newswire; no gold POS tags exist for resumes, so resume output "
                            "is compared with NLTK's perceptron tagger (agreement, not accuracy)."}
    (OUT / "hmm_report.json").write_text(json.dumps(report, indent=1))
    print("\nHeld-out treebank accuracy:", report["accuracy"])


if __name__ == "__main__":
    main()
