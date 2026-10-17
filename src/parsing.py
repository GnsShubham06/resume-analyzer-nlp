"""Practical 5: POS tagging, shallow parsing (chunking), deep (dependency)
parsing and CFG parse trees on real resume sentences.

Run: python -m src.parsing
"""
from __future__ import annotations

import json

import nltk
import pandas as pd

from .config import DATASET_CSV, PLOTS_DIR, RESULTS_DIR
from .nlp_utils import ensure_nltk, get_spacy
from .preprocessing import clean_text, split_sentences

OUT = RESULTS_DIR / "practical5_parsing"

CHUNK_GRAMMAR = r"""
  NP: {<DT|PRP\$>?<JJ.*|VBN|VBG>*<NN.*>+}      # noun phrase, e.g. "employee benefit programs"
  PP: {<IN><NP>}                              # prepositional phrase
  VP: {<VB.*><RB.*>?<NP|PP>+}                 # verb phrase with its object(s)
"""

# A CFG whose terminals are Penn-Treebank POS tags.  Parsing the *tag
# sequence* lets a small hand-written grammar cover real resume sentences.
POS_CFG = nltk.CFG.fromstring("""
  S    -> VP | NP VP | NP | VP CC VP | NP VP PUNCT | VP PUNCT | NP PUNCT
  VP   -> V NP | V NP PP | V PP | V | V NP PP PP | V ADJP | V NP NP | V S | RB VP
  NP   -> DET NOM | NOM | NP PP | NP CC NP | PRP | NP COMMA NP | CD NOM | DET ADJP NOM
  NOM  -> N | ADJP NOM | N NOM | VBG NOM | VBN NOM
  ADJP -> JJ | JJ ADJP | RB JJ
  PP   -> P NP
  V    -> 'VB' | 'VBD' | 'VBZ' | 'VBP' | 'VBN' | 'VBG' | 'MD' V
  N    -> 'NN' | 'NNS' | 'NNP' | 'NNPS'
  DET  -> 'DT' | 'PRP$'
  P    -> 'IN' | 'TO'
  JJ   -> 'JJ' | 'JJR' | 'JJS'
  RB   -> 'RB'
  CC   -> 'CC'
  CD   -> 'CD'
  PRP  -> 'PRP'
  VBG  -> 'VBG'
  VBN  -> 'VBN'
  COMMA -> ','
  PUNCT -> '.'
""")


def pick_sentences(df: pd.DataFrame, n: int = 6, min_words: int = 5, max_words: int = 14) -> list[tuple]:
    """Short, real, full sentences from different resumes."""
    out = []
    for _, r in df.iterrows():
        for s in split_sentences(clean_text(r["anonymized_text"])):
            w = s.split()
            if min_words <= len(w) <= max_words and s[0].isupper() and s.endswith(".") \
                    and "[" not in s and not s.isupper():
                out.append((r["resume_id"], s))
                break
        if len(out) >= n:
            break
    return out


def pos_tags(sentence: str) -> dict:
    ensure_nltk()
    toks = nltk.word_tokenize(sentence)
    nltk_tags = nltk.pos_tag(toks)
    doc = get_spacy()(sentence)
    return {"nltk": nltk_tags, "spacy": [(t.text, t.tag_, t.pos_) for t in doc]}


def shallow_parse(sentence: str):
    ensure_nltk()
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    tree = nltk.RegexpParser(CHUNK_GRAMMAR).parse(tagged)
    chunks = [(st.label(), " ".join(w for w, _ in st.leaves())) for st in tree.subtrees()
              if st.label() in ("NP", "VP", "PP")]
    # IOB tags need a flat (one-level) tree -> NP-only chunker
    np_tree = nltk.RegexpParser(CHUNK_GRAMMAR.strip().splitlines()[0]).parse(tagged)
    iob = nltk.chunk.tree2conlltags(np_tree)
    return tree, chunks, iob


def dependency_parse(sentence: str) -> list[dict]:
    doc = get_spacy()(sentence)
    return [{"token": t.text, "dep": t.dep_, "head": t.head.text, "pos": t.pos_,
             "children": [c.text for c in t.children]} for t in doc]


def save_dependency_svg(sentence: str, fname: str):
    from spacy import displacy
    svg = displacy.render(get_spacy()(sentence), style="dep", jupyter=False,
                          options={"compact": True, "distance": 95})
    (PLOTS_DIR / fname).write_text(svg, encoding="utf-8")
    try:  # PNG copy for Word/PDF reports (optional dependency)
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(PLOTS_DIR / fname.replace(".svg", ".png")),
                         output_width=1400, background_color="white")
    except Exception:
        pass


def cfg_parse_pos(sentence: str):
    """Parse the POS-tag sequence of a real sentence with POS_CFG."""
    ensure_nltk()
    tags = [t for _, t in nltk.pos_tag(nltk.word_tokenize(sentence))]
    words = nltk.word_tokenize(sentence)
    parser = nltk.ChartParser(POS_CFG)
    try:
        trees = list(parser.parse(tags))
    except ValueError as e:  # a tag not covered by the grammar
        return None, tags, str(e)
    if not trees:
        return None, tags, "no parse (sentence structure not covered by the small grammar)"
    tree = trees[0]
    # put the words back under the POS pre-terminals for readability
    leaves = iter(words)
    for pos in tree.treepositions("leaves"):
        tree[pos] = f"{tree[pos]}:{next(leaves)}"
    return tree, tags, f"{len(trees)} parse(s) found"


def cfg_parse_lexical():
    """Classic word-level CFG built for one real resume sentence."""
    grammar = nltk.CFG.fromstring("""
      S   -> VP
      VP  -> V NP | V NP PP
      NP  -> N | ADJ N | N N | NP CC NP | ADJ N N
      PP  -> P NP
      V   -> 'maintain' | 'calibrate'
      CC  -> 'and'
      ADJ -> 'test'
      N   -> 'instruments' | 'testing' | 'capabilities'
      P   -> 'to'
    """)
    sentence = "maintain and calibrate test instruments"   # from ENG_01: "Maintain and calibrate test instruments ..."
    grammar2 = nltk.CFG.fromstring("""
      S  -> VP
      VP -> V NP | VP CC VP | V
      NP -> ADJ N
      V  -> 'maintain' | 'calibrate'
      CC -> 'and'
      ADJ -> 'test'
      N  -> 'instruments'
    """)
    trees = list(nltk.ChartParser(grammar2).parse(sentence.split()))
    return sentence, trees


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    sents = pick_sentences(df)
    report = {"sentences": [], "cfg_pos_coverage": None}
    lines = []
    for i, (rid, s) in enumerate(sents):
        tree, chunks, iob = shallow_parse(s)
        dep = dependency_parse(s)
        cfg_tree, tags, status = cfg_parse_pos(s)
        report["sentences"].append({"resume_id": rid, "sentence": s, "pos": pos_tags(s),
                                    "chunks": chunks, "iob": iob, "dependency": dep,
                                    "cfg_status": status,
                                    "cfg_tree": cfg_tree.pformat(margin=120) if cfg_tree else None})
        lines += [f"\n=== [{rid}] {s}", "POS (NLTK): " + " ".join(f"{w}/{t}" for w, t in pos_tags(s)["nltk"]),
                  "Chunks: " + "; ".join(f"[{l} {t}]" for l, t in chunks),
                  "Chunk tree: " + tree.pformat(margin=200),
                  "Dependencies: " + ", ".join(f"{d['token']}<-{d['dep']}-{d['head']}" for d in dep),
                  f"CFG over POS tags: {status}"]
        if cfg_tree:
            lines.append(cfg_tree.pformat(margin=100))
        if i == 0:
            save_dependency_svg(s, "dependency_parse_example.svg")
    # coverage of the POS-level CFG over many real sentences
    sample = []
    for _, r in df.iterrows():
        sample += [s for s in split_sentences(clean_text(r["anonymized_text"])) if 4 <= len(s.split()) <= 15][:6]
    ok = sum(cfg_parse_pos(s)[0] is not None for s in sample)
    report["cfg_pos_coverage"] = {"sentences_tried": len(sample), "parsed": ok,
                                  "coverage_pct": round(100 * ok / max(len(sample), 1), 1)}
    sent, trees = cfg_parse_lexical()
    report["lexical_cfg"] = {"sentence": sent, "n_parses": len(trees),
                             "trees": [t.pformat() for t in trees]}
    lines += [f"\n=== Word-level CFG for a real phrase from ENG_01: '{sent}' -> {len(trees)} parse(s)"]
    lines += [t.pformat() for t in trees]
    lines.append(f"\nPOS-level CFG coverage on {len(sample)} real resume sentences: {ok} parsed "
                 f"({report['cfg_pos_coverage']['coverage_pct']}%)")
    (OUT / "parsing_report.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "parsing_output.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
