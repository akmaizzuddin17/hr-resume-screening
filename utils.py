from pypdf import PdfReader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import uuid
import os
import io
import re
import json
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_REGION     = os.getenv("PINECONE_REGION")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# ── Key helpers ────────────────────────────────────────────────────────────────

def _sanitize_api_key(key: str | None) -> str | None:
    if not key:
        return None
    key = key.strip()
    if (key.startswith("'") and key.endswith("'")) or (key.startswith('"') and key.endswith('"')):
        key = key[1:-1].strip()
    return key or None


def _get_api_key(provided: str | None) -> str:
    key = _sanitize_api_key(provided or os.getenv("PINECONE_API_KEY"))
    if not key:
        raise RuntimeError(
            "Pinecone API key not found. Set PINECONE_API_KEY in your .env file."
        )
    return key


# ── PDF extraction ─────────────────────────────────────────────────────────────

def get_pdf_text(pdf_doc) -> str:
    """
    Extract all text from a PDF.
    Wraps Streamlit UploadedFile in BytesIO so pypdf always gets a
    fresh, seekable stream (fixes the empty page_content bug).
    """
    if hasattr(pdf_doc, "read"):
        raw_bytes = pdf_doc.read()
        if hasattr(pdf_doc, "seek"):
            pdf_doc.seek(0)
        stream = io.BytesIO(raw_bytes)
    else:
        stream = pdf_doc

    text_parts = []
    try:
        reader = PdfReader(stream)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted.strip())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PDF '{getattr(pdf_doc, 'name', pdf_doc)}': {exc}"
        ) from exc

    return "\n".join(text_parts)


def create_docs(user_pdf_list, unique_id: str) -> list[Document]:
    docs = []
    for filename in user_pdf_list:
        text = get_pdf_text(filename)
        if not text.strip():
            text = "[EMPTY — no extractable text found in this PDF. It may be scanned/image-based.]"
        docs.append(Document(
            page_content=text,
            metadata={
                "name":      filename.name,
                "id":        uuid.uuid4().hex,
                "type":      filename.type,
                "size":      filename.size,
                "unique_id": unique_id,
            }
        ))
    return docs


# ── Embedding ──────────────────────────────────────────────────────────────────

def create_embedding_instance() -> HuggingFaceEmbeddings:
    """
    all-MiniLM-L6-v2 → 384-dim vectors.
    Pinecone index MUST be created with dimension=384.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Pinecone helpers ───────────────────────────────────────────────────────────

def _build_vector_store(index_name: str, embedding: HuggingFaceEmbeddings) -> PineconeVectorStore:
    return PineconeVectorStore(index_name=index_name, embedding=embedding)


def push_to_pinecone(pinecone_apikey, pinecone_env, index_name, embedding, docs):
    api_key = _get_api_key(pinecone_apikey)
    os.environ["PINECONE_API_KEY"] = api_key
    vs = _build_vector_store(index_name, embedding)
    vs.add_documents(docs)


def fetch_from_pinecone(
    pinecone_apikey, pinecone_env, index_name, embedding, query, k, unique_id
) -> list[tuple[Document, float]]:
    api_key = _get_api_key(pinecone_apikey)
    os.environ["PINECONE_API_KEY"] = api_key
    vs = _build_vector_store(index_name, embedding)

    raw = vs.similarity_search_with_score(
        query=query,
        k=k,
        filter={"unique_id": unique_id},
    )

    normalized: list[tuple[Document, float | None]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            normalized.append((item[0], item[1]))
        elif isinstance(item, dict):
            doc = item.get("document") or Document(
                page_content=item.get("page_content") or item.get("text") or "",
                metadata=item.get("metadata") or {},
            )
            score = item.get("score") or item.get("distance")
            normalized.append((doc, score))
        else:
            normalized.append((getattr(item, "document", item), getattr(item, "score", None)))

    normalized.sort(
        key=lambda p: (p[1] is not None, p[1] if p[1] is not None else -1),
        reverse=True,
    )
    return normalized  # type: ignore[return-value]


# ── Local highlight analysis (no external API) ─────────────────────────────────

# Common English stop-words to filter out from keyword extraction
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "that","this","these","those","it","its","we","our","you","your","they",
    "their","he","his","she","her","i","my","me","us","as","if","by","from",
    "up","about","into","through","during","including","until","against",
    "between","while","than","so","yet","both","either","each","more","most",
    "other","some","such","no","not","only","own","same","then","when","where",
    "how","all","any","few","more","also","just","because","over","under",
    "again","further","once","very","here","there","what","which","who","whom",
    "based","using","across","within","new","good","well","work","use","make",
    "provide","ensure","support","required","experience","ability","strong",
    "excellent","knowledge","understanding","skills","skill","role","team",
    "position","job","candidate","applicant","preferred","plus","bonus",
    "minimum","years","year","months","month","time","day","days","per",
}


def _extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Extract meaningful keywords from text using frequency + length heuristic.
    Returns a de-duped list of the top_n most significant terms.
    """
    # Lowercase, keep only alphabetic tokens of length >= 3
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS]

    # Count frequencies
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # Sort by frequency descending, break ties by length (longer = more specific)
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [word for word, _ in ranked[:top_n]]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristic."""
    # Split on period/exclamation/question mark followed by whitespace or end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on newlines that look like section breaks
    sentences = []
    for chunk in raw:
        for line in chunk.split('\n'):
            line = line.strip()
            if len(line) > 30:   # ignore very short fragments
                sentences.append(line)
    return sentences


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure-Python cosine similarity (no extra dependencies)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def analyse_resume_match(
    resume_text: str,
    job_description: str,
    embedding: HuggingFaceEmbeddings,
    top_sentences: int = 5,
) -> dict:
    """
    Fully local analysis — uses only HuggingFace embeddings.

    Returns:
        summary           – one-line match summary string
        matched_keywords  – JD keywords found in the resume
        missing_keywords  – JD keywords absent from the resume
        strengths         – top-scoring sentences rephrased as strength bullets
        gaps              – missing keyword groups phrased as gap bullets
        highlighted_sentences – list of raw resume sentences most relevant to JD
    """
    if resume_text.startswith("[EMPTY"):
        return {
            "summary": "Could not analyse — no text was extracted from this PDF.",
            "matched_keywords": [],
            "missing_keywords": [],
            "strengths": [],
            "gaps": ["No readable text found in this PDF (possibly scanned)."],
            "highlighted_sentences": [],
        }

    resume_lower = resume_text.lower()

    # ── 1. Keyword matching ────────────────────────────────────────────────────
    jd_keywords   = _extract_keywords(job_description, top_n=40)
    matched   = [kw for kw in jd_keywords if re.search(rf'\b{re.escape(kw)}\b', resume_lower)]
    missing   = [kw for kw in jd_keywords if kw not in matched]

    # Cap lists for display
    matched  = matched[:12]
    missing  = missing[:8]

    # ── 2. Sentence relevance via embeddings ───────────────────────────────────
    sentences = _split_sentences(resume_text[:5000])   # cap to first 5000 chars

    highlighted: list[str] = []
    sentence_scores: list[tuple[str, float]] = []

    if sentences:
        try:
            jd_vec      = embedding.embed_query(job_description[:1000])
            sent_vecs   = embedding.embed_documents(sentences)

            for sent, vec in zip(sentences, sent_vecs):
                sim = _cosine_similarity(jd_vec, vec)
                sentence_scores.append((sent, sim))

            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            highlighted = [s for s, _ in sentence_scores[:top_sentences]]
        except Exception:
            # If embedding fails for any reason, skip highlights gracefully
            highlighted = []

    # ── 3. Build human-readable summary ───────────────────────────────────────
    match_pct = int(len(matched) / max(len(jd_keywords[:40]), 1) * 100)
    if match_pct >= 70:
        fit = "strong fit"
    elif match_pct >= 45:
        fit = "moderate fit"
    elif match_pct >= 20:
        fit = "partial fit"
    else:
        fit = "low fit"

    summary = (
        f"This resume is a {fit} for the role — "
        f"{len(matched)} of the top {min(len(jd_keywords),40)} job keywords were found "
        f"({match_pct}% keyword coverage). "
        f"{len(missing)} important requirement(s) appear to be missing."
    )

    # ── 4. Strengths — top scoring sentences as bullets ───────────────────────
    strengths = []
    for sent, sim in sentence_scores[:3]:
        short = sent[:120].rstrip() + ("…" if len(sent) > 120 else "")
        strengths.append(f"{short}  ·  (relevance {sim:.2f})")

    # ── 5. Gaps — missing keyword groups ──────────────────────────────────────
    gaps = []
    if missing:
        gaps.append(f"Missing keywords from JD: {', '.join(missing[:5])}")
    if len(missing) > 5:
        gaps.append(f"Additional gaps: {', '.join(missing[5:])}")
    if not gaps:
        gaps.append("No significant keyword gaps detected.")

    return {
        "summary":             summary,
        "matched_keywords":    matched,
        "missing_keywords":    missing,
        "strengths":           strengths,
        "gaps":                gaps,
        "highlighted_sentences": highlighted,
    }
