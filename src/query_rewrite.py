import os
import re
from typing import List, Dict

try:
    import google.generativeai as genai
except Exception:
    genai = None


# Simple abbreviation expansion map (extendable)
ABBREV_MAP = {
    "ot": "operational technology",
    "ics": "industrial control systems",
    "cisa": "CISA",
    "nist": "NIST",
}

# Small synonym candidates to produce alternative formulations
SYNONYMS = {
    "remote access": ["remote-access", "remote connectivity", "remote management"],
    "vulnerability": ["vulnerability", "security flaw", "weakness"],
    "firewall": ["network firewall", "perimeter firewall"],
}


def _canonicalize(query: str) -> str:
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _expand_abbrev(query: str) -> str:
    if not ABBREV_MAP:
        return query
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in ABBREV_MAP.keys()) + r")\b", re.I)
    return pattern.sub(lambda m: ABBREV_MAP[m.group(0).lower()], query)


def _synonym_variants(query: str) -> List[str]:
    variants: List[str] = []
    lower = query.lower()
    for key, subs in SYNONYMS.items():
        if key in lower:
            for s in subs:
                variants.append(re.sub(re.escape(key), s, query, flags=re.I))
    return variants


def _rewrite_with_llm(query: str, max_outputs: int = 3, model_name: str = "gemini-2.5-flash") -> List[str]:
    """
    Optional LLM-based paraphrasing (uses google.genai if available and GEMINI_API_KEY present).
    Falls back to empty list if GenAI is not configured.
    """
    if genai is None or not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return []

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        prompt = (
            "Paraphrase the following question into concise alternative formulations (preserve intent).\n"
            f"Question: {query}\n\nProvide {max_outputs} numbered alternatives."
        )
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        text = getattr(response, "text", "") or ""
        lines = [ln.strip() for ln in re.split(r"\n|\r", text) if ln.strip()]
        # remove numbering tokens if present
        cleaned = []
        for ln in lines:
            ln = re.sub(r"^\s*\d+[\).\-]\s*", "", ln)
            ln = re.sub(r"^\s*[-*]\s*", "", ln)
            if ln:
                cleaned.append(ln)
            if len(cleaned) >= max_outputs:
                break
        return cleaned
    except Exception:
        return []


def rewrite_query(query: str, max_candidates: int = 6) -> List[Dict[str, str]]:
    """Return a list of rewrite candidates with a simple `type` label and `text`.

    The function tries these strategies (in order): canonicalize, expand
    abbreviations, synonym-based variants, optional LLM paraphrases.
    """
    candidates: List[Dict[str, str]] = []
    q0 = _canonicalize(query)
    candidates.append({"type": "canonical", "text": q0})

    q1 = _expand_abbrev(q0)
    if q1 != q0:
        candidates.append({"type": "expand_abbrev", "text": q1})

    for v in _synonym_variants(q1):
        candidates.append({"type": "synonym_variant", "text": v})

    # Try LLM-based paraphrase (best-effort; may return empty list)
    for i, v in enumerate(_rewrite_with_llm(q0, max_outputs=3)):
        candidates.append({"type": "llm_paraphrase", "text": v})

    # Deduplicate while preserving order and limit
    seen = set()
    out: List[Dict[str, str]] = []
    for c in candidates:
        t = c["text"].strip()
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": c["type"], "text": t})
        if len(out) >= max_candidates:
            break

    return out
