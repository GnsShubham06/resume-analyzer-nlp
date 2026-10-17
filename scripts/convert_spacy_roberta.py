"""Build a local Hugging Face RoBERTa checkpoint when the HF Hub is blocked.

WHY: the machine used to produce the submitted results could not reach
huggingface.co (HTTP 403 from the network proxy), so `distilbert-base-uncased`
could not be downloaded.  spaCy's `en_core_web_trf` 3.8.0 package (downloaded
from GitHub releases) ships a full pretrained RoBERTa-base encoder
(12 layers, 768 hidden, 125M parameters).  This script copies those weights
and the byte-level BPE vocabulary into the standard Hugging Face format so
that `src/train_transformer.py` can fine-tune it with the normal
`AutoModelForSequenceClassification` + `Trainer` code path.

Note: the encoder in en_core_web_trf was initialised from `roberta-base`
and further trained by Explosion on OntoNotes tagging/parsing/NER, so it is
a pretrained (not randomly initialised) language model.

In Google Colab you do NOT need this script: just use
`--model_name distilbert-base-uncased` (the default).

Usage:  python scripts/convert_spacy_roberta.py [--out models/pretrained/roberta-base-spacy]
Requires: pip install spacy spacy-curated-transformers, en_core_web_trf 3.8.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LOCAL_ROBERTA_DIR  # noqa: E402


def convert(out_dir: Path, verify: bool = True):
    import spacy
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    nlp = spacy.load("en_core_web_trf")
    trf_model = nlp.get_pipe("transformer").model
    curated, bpe = None, None
    for node in trf_model.walk():
        for shim in node.shims:
            curated = shim._model
        if node.name == "byte_bpe_encoder":
            bpe = node.attrs["byte_bpe_processor"]
    assert curated is not None and bpe is not None, "could not find transformer weights"
    sd = {k.replace("curated_encoder.", ""): v for k, v in curated.state_dict().items()}

    cfg = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
                        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                        max_position_embeddings=514, type_vocab_size=1, layer_norm_eps=1e-5,
                        pad_token_id=1, bos_token_id=0, eos_token_id=2)
    hf = {}
    e = "embeddings.inner."
    hf["embeddings.word_embeddings.weight"] = sd[e + "word_embeddings.weight"]
    hf["embeddings.position_embeddings.weight"] = sd[e + "position_embeddings.weight"]
    hf["embeddings.token_type_embeddings.weight"] = sd[e + "token_type_embeddings.weight"]
    hf["embeddings.LayerNorm.weight"] = sd[e + "layer_norm.weight"]
    hf["embeddings.LayerNorm.bias"] = sd[e + "layer_norm.bias"]
    for i in range(cfg.num_hidden_layers):
        c, h = f"layers.{i}.", f"encoder.layer.{i}."
        for p in ("weight", "bias"):
            q, k, v = sd[c + f"mha.input.{p}"].chunk(3, dim=0)   # fused [q;k;v]
            hf[h + f"attention.self.query.{p}"] = q
            hf[h + f"attention.self.key.{p}"] = k
            hf[h + f"attention.self.value.{p}"] = v
            hf[h + f"attention.output.dense.{p}"] = sd[c + f"mha.output.{p}"]
            hf[h + f"attention.output.LayerNorm.{p}"] = sd[c + f"attn_output_layernorm.{p}"]
            hf[h + f"intermediate.dense.{p}"] = sd[c + f"ffn.intermediate.{p}"]
            hf[h + f"output.dense.{p}"] = sd[c + f"ffn.output.{p}"]
            hf[h + f"output.LayerNorm.{p}"] = sd[c + f"ffn_output_layernorm.{p}"]

    model = RobertaModel(cfg, add_pooling_layer=False)
    missing, unexpected = model.load_state_dict({k: v.clone() for k, v in hf.items()}, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing and not unexpected, (missing, unexpected)

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = dict(sorted(bpe.vocab.items(), key=lambda kv: kv[1])) if isinstance(bpe.vocab, dict) else bpe.vocab
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    merges = bpe.merges
    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges:
            f.write((" ".join(m) if not isinstance(m, str) else m) + "\n")
    # build the fast tokenizer with the `tokenizers` library (works with
    # transformers v4 and v5), byte-level BPE exactly like RoBERTa
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    merges_list = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]
    core = Tokenizer(models.BPE(vocab=vocab, merges=merges_list))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    core.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0),
                                                       trim_offsets=True, add_prefix_space=False)
    tok = RobertaTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>",
                               sep_token="</s>", cls_token="<s>", unk_token="<unk>",
                               pad_token="<pad>", mask_token="<mask>", model_max_length=512)
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    report = {"source": "spaCy en_core_web_trf 3.8.0 RoBERTa-base encoder",
              "n_parameters": sum(p.numel() for p in model.parameters()),
              "vocab_size": len(vocab)}
    if verify:
        text = "Experienced registered nurse skilled in patient care and EMR documentation."
        ids_hf = tok(text)["input_ids"]
        ids_sp = [0] + bpe.encode_as_ids(text) + [2]
        report["tokenizer_ids_match"] = ids_hf == ids_sp
        model.eval()
        curated.eval()   # disable dropout in the spaCy copy before comparing
        try:  # numerical check: both encoders must give the same hidden states
            ids = torch.tensor([ids_sp])
            with torch.no_grad():
                h_hf = model(input_ids=ids).last_hidden_state
                h_sp = curated.curated_encoder(ids).last_hidden_layer_states
            report["max_abs_diff_hidden_states"] = float((h_hf - h_sp).abs().max())
        except Exception as ex:  # API differences between curated-transformers versions
            report["numerical_check"] = f"skipped: {ex}"
    (out_dir / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LOCAL_ROBERTA_DIR))
    convert(Path(ap.parse_args().out))
