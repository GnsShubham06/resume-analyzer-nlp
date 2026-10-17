"""Streamlit demo: Resume Analyzer Using NLP.

Run:  streamlit run app.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import RESULTS_DIR  # noqa: E402
from src.predict import CLASSIFIER_WARNING, analyze_file, analyze_text, find_transformer_dir  # noqa: E402

st.set_page_config(page_title="Resume Analyzer (NLP)", layout="wide")
st.title("Resume Analyzer Using NLP")
st.caption("Task A: domain classification (Engineering / HR / Finance / Healthcare / Agriculture) · "
           "Task B: skill & information extraction")

tdir = find_transformer_dir()
st.sidebar.header("Models")
st.sidebar.write("Fine-tuned transformer:", f"`{tdir.name}`" if tdir else "not trained yet")
st.sidebar.write("Baseline:", "TF-IDF + Logistic Regression")
use_trf = st.sidebar.checkbox("Use transformer", value=tdir is not None, disabled=tdir is None)
comparison = RESULTS_DIR / "model_comparison.csv"
if comparison.exists():
    st.sidebar.subheader("Saved evaluation (same test resumes)")
    cmp = pd.read_csv(comparison)
    for r in cmp[(cmp["split"] == "test") & (cmp["resumes"] == "all")].itertuples():
        st.sidebar.write(f"{r.model}: accuracy {r.accuracy:.2f}, macro-F1 {r.macro_f1:.2f} (n={r.n_samples})")

st.warning(CLASSIFIER_WARNING)

up = st.file_uploader("Upload a resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
pasted = st.text_area("...or paste resume text", height=120)

if st.button("Analyze", type="primary") and (up or pasted.strip()):
    with st.spinner("Extracting, cleaning and analysing..."):
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                tmp.write(up.getvalue())
            res = analyze_file(tmp.name, use_transformer=use_trf)
            res["file"] = up.name
        else:
            res = analyze_text(pasted, use_transformer=use_trf)
    if res.get("status") != "ok":
        st.error(f"Could not read the file: {res.get('status')} - {res.get('error')}")
        st.stop()

    c = res["classification"]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted domain")
        if c.get("transformer"):
            t = c["transformer"]
            st.metric("Fine-tuned transformer", t["predicted"], f"{max(t['probabilities'].values()):.0%} confidence")
            st.bar_chart(pd.Series(t["probabilities"]))
        elif c.get("transformer_error"):
            st.error(c["transformer_error"])
        if c.get("baseline"):
            b = c["baseline"]
            st.metric("TF-IDF baseline", b["predicted"], f"{max(b['probabilities'].values()):.0%} confidence")
    with col2:
        st.subheader("Extracted skills")
        if res["skills"]:
            st.dataframe(pd.DataFrame(res["skills"]), hide_index=True, width="stretch")
        st.write("**Lexicon domain evidence:**", res["lexicon_domain_scores"])

    st.subheader("Information extraction")
    a, b, c3 = st.columns(3)
    a.write("**Job titles**")
    a.write(res["job_titles"])
    b.write("**Education**")
    b.json(res["education"])
    c3.write("**Experience**")
    c3.json(res["experience"])
    st.write("**Organizations (spaCy NER + institution patterns):**", ", ".join(res["organizations"]) or "-")

    st.subheader("Named entities")
    e1, e2 = st.columns(2)
    e1.write("Pretrained spaCy NER")
    e1.dataframe(pd.DataFrame(res["entities"]["pretrained_spacy"]), hide_index=True, width="stretch")
    e2.write("Rule-based (skills, degrees, titles, dates)")
    e2.dataframe(pd.DataFrame(res["entities"]["rule_based"]), hide_index=True, width="stretch")

    with st.expander("Cleaned & anonymized text"):
        st.text(res["cleaned_text_preview"])
        st.write("Anonymization:", res["anonymization"], "Language:", res["language_validation"])

    st.download_button("Download analysis (JSON)", json.dumps(res, indent=1, default=str),
                       file_name=f"analysis_{Path(res['file']).stem}.json", mime="application/json")
