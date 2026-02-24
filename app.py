import time
import uuid

import streamlit as st

from utils import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    create_docs,
    create_embedding_instance,
    fetch_from_pinecone,
    push_to_pinecone,
    analyse_resume_match,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🎯",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }

.stApp {
    background: #07070f;
    color: #dde1f0;
}

/* Hero */
.hero { text-align:center; padding:2.6rem 1rem 0.4rem; }
.hero h1 {
    font-size:2.8rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(100deg,#c084fc,#818cf8,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0.3rem;
}
.hero p { color:#4b5563; font-weight:300; font-size:1rem; }

/* Section labels */
.slabel {
    font-size:0.68rem; font-weight:700; letter-spacing:0.14em;
    text-transform:uppercase; color:#374151; margin-bottom:0.45rem;
}

/* Fancy divider */
.fdiv {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(129,140,248,0.25),transparent);
    margin:1.6rem 0;
}

/* Result card */
.rcard {
    background:#0d0d1a;
    border:1px solid #16162a;
    border-radius:18px;
    padding:1.5rem 1.7rem 1.2rem;
    margin-bottom:0.6rem;
    position:relative; overflow:hidden;
}
.rcard::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--bar);
}

/* Rank badge */
.rbadge {
    width:44px; height:44px; border-radius:50%;
    background:var(--bar);
    display:inline-flex; align-items:center; justify-content:center;
    font-size:1.05rem; font-weight:800; color:#fff; flex-shrink:0;
}

/* Score bar */
.strack { height:8px; background:#111126; border-radius:4px; overflow:hidden; margin-top:7px; }
.sfill  { height:100%; border-radius:4px; background:var(--bar); }

/* Score number */
.snum {
    font-family:'JetBrains Mono',monospace;
    font-size:1.5rem; font-weight:700;
}

/* Keyword pills */
.pills { display:flex; flex-wrap:wrap; gap:5px; margin-top:5px; }
.pill  { display:inline-block; padding:3px 9px; border-radius:20px;
         font-size:0.74rem; font-weight:600; letter-spacing:0.02em; }
.pmatch { background:rgba(52,211,153,.11); border:1px solid rgba(52,211,153,.3); color:#34d399; }
.pmiss  { background:rgba(248,113,113,.10); border:1px solid rgba(248,113,113,.28); color:#f87171; }

/* Summary box */
.sumbox {
    background:rgba(129,140,248,.07); border-left:3px solid #818cf8;
    border-radius:0 10px 10px 0; padding:0.75rem 1rem;
    font-size:0.88rem; color:#c7d2fe; line-height:1.7; margin-bottom:1.1rem;
}

/* Bullet lists */
.blist { list-style:none; padding:0; margin:0; }
.blist li { padding:4px 0 4px 22px; position:relative;
            font-size:0.85rem; color:#9ca3af; line-height:1.5; }
.blist li::before { position:absolute; left:0; font-weight:700; }
.slist li::before { content:"✦"; color:#34d399; }
.glist li::before { content:"△"; color:#f87171; }

/* Highlighted resume box */
.hlbox {
    background:#050510; border:1px solid #16162a; border-radius:12px;
    padding:1rem 1.2rem;
    font-family:'JetBrains Mono',monospace; font-size:0.79rem;
    line-height:1.85; color:#6b7280;
    max-height:280px; overflow-y:auto;
    white-space:pre-wrap; word-break:break-word;
}
/* highlighted sentence */
.hl {
    background:rgba(251,191,36,.18); color:#fde68a;
    border-radius:3px; padding:1px 4px;
}

/* Streamlit widget overrides */
.stTextArea textarea {
    background:#0d0d1a !important; border:1px solid #16162a !important;
    border-radius:12px !important; color:#dde1f0 !important;
    font-family:'Sora',sans-serif !important;
}
.stTextArea textarea:focus {
    border-color:#818cf8 !important;
    box-shadow:0 0 0 2px rgba(129,140,248,.18) !important;
}
.stFileUploader {
    border:2px dashed rgba(129,140,248,.22) !important;
    border-radius:14px !important; background:#0d0d1a !important;
}
div.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#2563eb) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; font-size:1rem !important;
    padding:0.7rem 2rem !important; width:100% !important;
    letter-spacing:0.02em !important;
}
div.stButton > button:hover { opacity:0.87 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = ""


# ── Colour palette per score ───────────────────────────────────────────────────
def palette(score: float) -> dict:
    if score >= 0.75:
        return {"bar": "linear-gradient(90deg,#34d399,#10b981)",
                "text": "#34d399", "label": "Excellent match", "emoji": "🟢"}
    elif score >= 0.50:
        return {"bar": "linear-gradient(90deg,#60a5fa,#3b82f6)",
                "text": "#60a5fa", "label": "Good match",      "emoji": "🔵"}
    elif score >= 0.30:
        return {"bar": "linear-gradient(90deg,#fbbf24,#f59e0b)",
                "text": "#fbbf24", "label": "Partial match",   "emoji": "🟡"}
    else:
        return {"bar": "linear-gradient(90deg,#f87171,#ef4444)",
                "text": "#f87171", "label": "Low match",       "emoji": "🔴"}


def apply_highlights(text: str, sentences: list[str]) -> str:
    """Wrap matched sentences in <span class='hl'>…</span>."""
    for sent in sentences:
        s = sent.strip()
        if s and s in text:
            text = text.replace(s, f"<span class='hl'>{s}</span>", 1)
    return text


# ── App ────────────────────────────────────────────────────────────────────────
def main() -> None:

    st.markdown("""
    <div class="hero">
        <h1>🎯 Resume Screening AI</h1>
        <p>Semantic search + local AI analysis — ranked candidates with intelligent highlights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Input columns
    col_jd, col_up = st.columns([3, 2], gap="large")

    with col_jd:
        st.markdown('<div class="slabel">📋 Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "jd", label_visibility="collapsed",
            placeholder="Paste the full job description here…",
            height=240, key="job_desc",
        )

    with col_up:
        st.markdown('<div class="slabel">📎 Upload Resumes (PDF)</div>', unsafe_allow_html=True)
        pdf_files = st.file_uploader(
            "pdfs", label_visibility="collapsed",
            type=["pdf"], accept_multiple_files=True, key="pdf_upload",
        )
        if pdf_files:
            for f in pdf_files:
                st.markdown(
                    f'<div style="font-size:.82rem;color:#6b7280;margin:2px 0;">'
                    f'📄 {f.name} <span style="color:#374151">({round(f.size/1024,1)} KB)</span></div>',
                    unsafe_allow_html=True,
                )
        st.write("")
        st.markdown('<div class="slabel">🔢 Top N results</div>', unsafe_allow_html=True)
        k_value = st.slider("k", min_value=1, max_value=10, value=3, label_visibility="collapsed")

    st.write("")
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        submit = st.button("🔍 Screen Resumes")

    # Validation
    if not submit:
        return
    if not job_description.strip():
        st.warning("⚠️  Please paste a job description first.")
        return
    if not pdf_files:
        st.warning("⚠️  Please upload at least one PDF resume.")
        return
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        st.error("❌  Pinecone config missing — check PINECONE_API_KEY and PINECONE_INDEX_NAME in .env")
        return

    # Pipeline
    bar = st.progress(0, text="Starting…")
    try:
        bar.progress(10, text="📄 Reading PDFs…")
        st.session_state["unique_id"] = uuid.uuid4().hex
        docs = create_docs(pdf_files, st.session_state["unique_id"])

        empty_names = [d.metadata["name"] for d in docs if d.page_content.startswith("[EMPTY")]
        if empty_names:
            st.warning(f"⚠️  No text extracted from: {', '.join(empty_names)} (scanned PDF?).")

        bar.progress(25, text="🧠 Loading embedding model…")
        embedding = create_embedding_instance()

        bar.progress(42, text="☁️  Uploading to Pinecone…")
        push_to_pinecone(
            pinecone_apikey=PINECONE_API_KEY, pinecone_env=PINECONE_REGION,
            index_name=PINECONE_INDEX_NAME, embedding=embedding, docs=docs,
        )

        bar.progress(56, text="⏳ Waiting for Pinecone to index…")
        time.sleep(4)

        bar.progress(68, text="🔍 Semantic search…")
        results = fetch_from_pinecone(
            pinecone_apikey=PINECONE_API_KEY, pinecone_env=PINECONE_REGION,
            index_name=PINECONE_INDEX_NAME, embedding=embedding,
            query=job_description, k=k_value,
            unique_id=st.session_state["unique_id"],
        )

        # Local analysis per resume (uses existing embedding model — no new API)
        analyses = []
        total = len(results[:k_value])
        for idx, (doc, _) in enumerate(results[:k_value]):
            pct = 68 + int((idx / max(total, 1)) * 28)
            bar.progress(pct, text=f"📊 Analysing resume {idx+1}/{total}…")
            analyses.append(
                analyse_resume_match(
                    resume_text=doc.page_content,
                    job_description=job_description,
                    embedding=embedding,
                )
            )

        bar.progress(100, text="✅ Complete!")
        time.sleep(0.3)
        bar.empty()

    except Exception as exc:
        bar.empty()
        st.error(f"❌  {exc}")
        return

    if not results:
        st.warning("No results returned. Check your Pinecone index or try re-uploading.")
        return

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown('<div class="fdiv"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">'
        f'Top {min(k_value, len(results))} Candidate(s)</div>',
        unsafe_allow_html=True,
    )

    for rank, ((doc, score), analysis) in enumerate(zip(results[:k_value], analyses), start=1):
        meta    = doc.metadata if hasattr(doc, "metadata") else {}
        name    = meta.get("name", f"Resume {rank}")
        size_kb = round(meta.get("size", 0) / 1024, 1)
        score   = float(score) if score is not None else 0.0
        pal     = palette(score)
        pct     = int(score * 100)
        content = doc.page_content if hasattr(doc, "page_content") else ""

        matched   = analysis.get("matched_keywords", [])
        missing   = analysis.get("missing_keywords", [])
        strengths = analysis.get("strengths", [])
        gaps      = analysis.get("gaps", [])
        summary   = analysis.get("summary", "")
        hl_sents  = analysis.get("highlighted_sentences", [])

        # ── Score card ─────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="rcard" style="--bar:{pal['bar']};">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.1rem;">
                <div class="rbadge" style="background:{pal['bar']};">#{rank}</div>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                         title="{name}">📄 {name}</div>
                    <div style="font-size:0.78rem;color:#374151;">
                        {size_kb} KB &nbsp;·&nbsp;
                        <span style="color:{pal['text']};">{pal['emoji']} {pal['label']}</span>
                    </div>
                </div>
                <div style="text-align:right;min-width:72px;">
                    <div class="snum" style="color:{pal['text']};">{score:.3f}</div>
                    <div style="font-size:0.7rem;color:#374151;">similarity</div>
                </div>
            </div>
            <div class="strack"><div class="sfill" style="width:{pct}%;background:{pal['bar']};"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Detail expander ────────────────────────────────────────────────────
        with st.expander(f"📊 Full Analysis — {name}", expanded=(rank == 1)):

            # Summary
            if summary:
                st.markdown(f'<div class="sumbox">{summary}</div>', unsafe_allow_html=True)

            col_kw, col_sg = st.columns(2, gap="medium")

            with col_kw:
                # Matched
                st.markdown('<div class="slabel">✅ Matched Keywords</div>', unsafe_allow_html=True)
                if matched:
                    pills = "".join(f'<span class="pill pmatch">{k}</span>' for k in matched)
                    st.markdown(f'<div class="pills">{pills}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#374151;font-size:.84rem;">None detected</span>',
                                unsafe_allow_html=True)
                st.write("")

                # Missing
                st.markdown('<div class="slabel">❌ Missing Keywords</div>', unsafe_allow_html=True)
                if missing:
                    pills = "".join(f'<span class="pill pmiss">{k}</span>' for k in missing)
                    st.markdown(f'<div class="pills">{pills}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#374151;font-size:.84rem;">None identified</span>',
                                unsafe_allow_html=True)

            with col_sg:
                # Strengths
                st.markdown('<div class="slabel">💪 Top Matching Lines</div>', unsafe_allow_html=True)
                if strengths:
                    items = "".join(f"<li>{s}</li>" for s in strengths)
                    st.markdown(f'<ul class="blist slist">{items}</ul>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#374151;font-size:.84rem;">N/A</span>',
                                unsafe_allow_html=True)
                st.write("")

                # Gaps
                st.markdown('<div class="slabel">⚠️ Gaps</div>', unsafe_allow_html=True)
                if gaps:
                    items = "".join(f"<li>{g}</li>" for g in gaps)
                    st.markdown(f'<ul class="blist glist">{items}</ul>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#374151;font-size:.84rem;">None identified</span>',
                                unsafe_allow_html=True)

            # Highlighted resume text
            st.write("")
            st.markdown('<div class="slabel">📝 Resume Highlights</div>', unsafe_allow_html=True)
            if hl_sents:
                st.markdown(
                    '<div style="font-size:.76rem;color:#374151;margin-bottom:5px;">'
                    '🟡 Yellow sentences = highest semantic relevance to the job description</div>',
                    unsafe_allow_html=True,
                )
            preview  = content[:3500] + ("…" if len(content) > 3500 else "")
            hl_html  = apply_highlights(preview, hl_sents)
            st.markdown(f'<div class="hlbox">{hl_html}</div>', unsafe_allow_html=True)

        st.write("")

    # Debug
    with st.expander("🛠 Debug — raw scores"):
        st.json([
            {
                "rank":  i + 1,
                "file":  doc.metadata.get("name", "?") if hasattr(doc, "metadata") else "?",
                "score": round(float(s), 6) if s is not None else None,
                "chars": len(doc.page_content) if hasattr(doc, "page_content") else 0,
            }
            for i, (doc, s) in enumerate(results)
        ])


if __name__ == "__main__":
    main()
