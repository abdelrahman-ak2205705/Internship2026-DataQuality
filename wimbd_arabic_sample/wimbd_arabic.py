"""
WIMBD (What's In My Big Data?) — applied to an Arabic Common Crawl sample.

Reference: Elazar et al., "What's In My Big Data?", arXiv:2310.20707.

This script reproduces the spirit of the paper's analyses on a JSONL file
of Arabic CC records. Each line is expected to be:
    {"record": {"url", "normalized_url", "text", "timestamp",
                "content_length", "content_type", "language",
                "langdetect": {"lang", "score"}, "sinan_id"},
     "status": "PASSED" | ...}

What it produces in OUT_DIR:
    - wimbd_results.json   : raw structured results of every analysis
    - wimbd_dashboard.html : single self-contained Plotly dashboard
    - wimbd_report.md      : short textual summary

Analyses implemented (WIMBD §3-§5):
    1. Dataset statistics: doc / token / byte counts, length distributions
    2. URL / domain analysis: TLDs, top domains, top URLs
    3. Date-of-source distribution: per-year/month from CC timestamps
    4. Language identification distribution (langdetect field)
    5. Top n-grams (1/2/3-grams) on Arabic-normalized tokens
    6. Most-common documents (exact duplicates by content hash)
    7. Near-duplicate detection via MinHash LSH
    8. PII detection: emails, phone numbers, IPv4, URLs, credit-card-like
    9. Profanity / offensive content (Arabic blocklist)
   10. Self-contamination: longest repeated 50-grams across docs
   11. Quality signals: alpha-ratio, symbol-to-word ratio, mean word length
"""

from __future__ import annotations

import collections
import hashlib
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import regex
import tldextract
from datasketch import MinHash, MinHashLSH
from plotly.subplots import make_subplots

# -------------------------------------------------------------------- config
INPUT_PATH = Path(
    "/mnt/d/E/LLM/____CODES____/Omar/CC_Samples/samples_100k/state_output_sample1000.jsonl"
)
OUT_DIR = Path("/mnt/d/Python/wimbd_arabic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 30
NGRAM_TOP_K = 25
MINHASH_PERMS = 128
MINHASH_THRESHOLD = 0.8
CONTAM_NGRAM = 50      # n-gram size used for self-contamination search
CONTAM_MAX_DOCS = 1000 # cap docs scanned for contamination (full sample fits)


# ---------------------------------------------------- Arabic text utilities
# Strip tashkeel (diacritics) + tatweel; unify alef/ya/ta-marbuta forms.
_TASHKEEL = regex.compile(r"[ً-ٰٟـ]")
_ALEF_VARIANTS = regex.compile(r"[إأآا]")
_YA_VARIANTS = regex.compile(r"[ىي]")
_TA_MARBUTA = regex.compile(r"ة")
_ARABIC_LETTER = regex.compile(r"\p{Arabic}")
_WORD_RE = regex.compile(r"[\p{Arabic}A-Za-z0-9]+")

ARABIC_STOPWORDS = set("""
في من إلى على عن مع هذا هذه ذلك تلك التي الذي الذين هؤلاء أن إن كان كانت
يكون تكون قد لقد ما لا لم لن لن إذا إذ كل كلها كله بعض غير حيث حين بين فوق
تحت بعد قبل عند لدى لدي هو هي هم هن أنا أنت نحن أنتم أنتن هما أو أم ثم بل
حتى لكن أيضا أيضًا إنما إذن آخر أخرى الآن هنا هناك مثل بسبب لكي كي لذلك
هكذا هل ليس ليست ليسوا قال قالت قالوا يقول تقول كما به بها بهم له لها لهم
عليه عليها عليهم منه منها منهم فيه فيها فيهم وهو وهي وهم
""".split())

OFFENSIVE_WORDS = set("""
كلب حمار خنزير وسخ قذر تافه احمق غبي بليد لعنه لعنة جحيم قبيح كافر زنا
""".split())


def normalize_ar(text: str) -> str:
    text = _TASHKEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _YA_VARIANTS.sub("ي", text)
    text = _TA_MARBUTA.sub("ه", text)
    return text


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


# -------------------------------------------------------- PII regex patterns
PII_PATTERNS = {
    "email":   re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "url":     re.compile(r"https?://[^\s<>\"']+"),
    "ipv4":    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Phone numbers: Western (+...) plus Arabic-Indic-digit variants common in MENA.
    "phone":   regex.compile(r"(?<!\w)(?:\+?\d[\d \-٠-٩]{7,}\d)(?!\w)"),
    # Loose 13-19 digit run, often a card number; we don't validate Luhn.
    "card_like": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}


# ---------------------------------------------------------------- IO helpers
def stream_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


# ------------------------------------------------------------------ analysis
def analyze(records: list[dict]) -> dict[str, Any]:
    """Run every WIMBD-style analysis on the full in-memory list."""
    res: dict[str, Any] = {}

    # ---- §3.1 dataset statistics ----------------------------------------
    docs, tokens_per_doc, bytes_per_doc, chars_per_doc = [], [], [], []
    alpha_ratios, sym_word_ratios, mean_word_lens = [], [], []
    timestamps, content_types, statuses, langs, lang_scores = [], [], [], [], []
    urls, normalized_urls = [], []

    for r in records:
        rec = r.get("record", {})
        text = rec.get("text", "") or ""
        toks = tokenize(text)

        docs.append({
            "url": rec.get("url"),
            "normalized_url": rec.get("normalized_url"),
            "text": text,
            "timestamp": rec.get("timestamp"),
            "lang": (rec.get("langdetect") or {}).get("lang"),
            "lang_score": (rec.get("langdetect") or {}).get("score"),
            "content_type": rec.get("content_type"),
            "status": r.get("status"),
            "n_tokens": len(toks),
            "n_chars": len(text),
            "n_bytes": len(text.encode("utf-8", errors="ignore")),
        })

        tokens_per_doc.append(len(toks))
        chars_per_doc.append(len(text))
        bytes_per_doc.append(len(text.encode("utf-8", errors="ignore")))
        alpha = sum(c.isalpha() for c in text)
        alpha_ratios.append(alpha / max(1, len(text)))
        nonword = sum(1 for c in text if not c.isalnum() and not c.isspace())
        sym_word_ratios.append(nonword / max(1, len(toks)))
        mean_word_lens.append(np.mean([len(t) for t in toks]) if toks else 0.0)
        timestamps.append(rec.get("timestamp"))
        content_types.append(rec.get("content_type"))
        statuses.append(r.get("status"))
        ld = rec.get("langdetect") or {}
        langs.append(ld.get("lang"))
        lang_scores.append(ld.get("score"))
        urls.append(rec.get("url"))
        normalized_urls.append(rec.get("normalized_url"))

    res["totals"] = {
        "n_documents": len(docs),
        "n_tokens": int(sum(tokens_per_doc)),
        "n_chars": int(sum(chars_per_doc)),
        "n_bytes": int(sum(bytes_per_doc)),
        "n_unique_urls": len(set(u for u in urls if u)),
        "n_unique_normalized_urls": len(set(u for u in normalized_urls if u)),
    }
    res["length_stats"] = {
        "tokens_per_doc": _dist_stats(tokens_per_doc),
        "chars_per_doc":  _dist_stats(chars_per_doc),
        "bytes_per_doc":  _dist_stats(bytes_per_doc),
        "alpha_ratio":    _dist_stats(alpha_ratios),
        "symbol_per_word":_dist_stats(sym_word_ratios),
        "mean_word_len":  _dist_stats(mean_word_lens),
    }
    res["statuses"] = dict(collections.Counter(statuses))
    res["content_types"] = dict(collections.Counter(content_types).most_common(TOP_K))
    res["languages"] = dict(collections.Counter(langs).most_common(TOP_K))
    res["lang_score_stats"] = _dist_stats([s for s in lang_scores if s is not None])

    # ---- §3.2 URL / domain analysis -------------------------------------
    tld_counter, domain_counter, suffix_counter = (
        collections.Counter(), collections.Counter(), collections.Counter())
    for u in urls:
        if not u:
            continue
        ext = tldextract.extract(u)
        if ext.registered_domain:
            domain_counter[ext.registered_domain] += 1
        if ext.suffix:
            tld_counter[ext.suffix.split(".")[-1]] += 1
            suffix_counter[ext.suffix] += 1
    res["top_domains"]  = domain_counter.most_common(TOP_K)
    res["top_tlds"]     = tld_counter.most_common(TOP_K)
    res["top_suffixes"] = suffix_counter.most_common(TOP_K)
    url_counter = collections.Counter(u for u in urls if u)
    res["duplicate_urls"] = [
        (u, c) for u, c in url_counter.most_common(TOP_K) if c > 1
    ]

    # ---- §3.3 date-of-source distribution -------------------------------
    year_counter, month_counter = collections.Counter(), collections.Counter()
    for t in timestamps:
        if not t:
            continue
        m = re.match(r"(\d{4})-(\d{2})", t)
        if not m:
            continue
        year_counter[m.group(1)] += 1
        month_counter[f"{m.group(1)}-{m.group(2)}"] += 1
    res["by_year"]  = dict(sorted(year_counter.items()))
    res["by_month"] = dict(sorted(month_counter.items()))

    # ---- §4 n-gram analysis (after Arabic normalization + stopword strip)
    unigram_counter = collections.Counter()
    bigram_counter  = collections.Counter()
    trigram_counter = collections.Counter()
    word_lens_global = collections.Counter()
    type_token_seen: set[str] = set()
    for d in docs:
        toks = [t for t in tokenize(normalize_ar(d["text"]))]
        toks_nostop = [t for t in toks if t not in ARABIC_STOPWORDS and len(t) > 1]
        unigram_counter.update(toks_nostop)
        bigram_counter.update(zip(toks_nostop, toks_nostop[1:]))
        trigram_counter.update(zip(toks_nostop, toks_nostop[1:], toks_nostop[2:]))
        for t in toks_nostop:
            word_lens_global[len(t)] += 1
            type_token_seen.add(t)

    res["top_unigrams"] = [(w, c) for w, c in unigram_counter.most_common(NGRAM_TOP_K)]
    res["top_bigrams"]  = [(" ".join(g), c) for g, c in bigram_counter.most_common(NGRAM_TOP_K)]
    res["top_trigrams"] = [(" ".join(g), c) for g, c in trigram_counter.most_common(NGRAM_TOP_K)]
    res["vocab"] = {
        "n_types": len(unigram_counter),
        "n_unique_after_norm": len(type_token_seen),
        "word_length_distribution": dict(sorted(word_lens_global.items())),
    }

    # ---- §5.1 exact-duplicate documents ---------------------------------
    fp_counter: collections.Counter[str] = collections.Counter()
    fp_examples: dict[str, dict] = {}
    for d in docs:
        fp = fingerprint(d["text"])
        fp_counter[fp] += 1
        fp_examples.setdefault(fp, d)
    dup_groups = [(fp, c) for fp, c in fp_counter.most_common() if c > 1]
    res["exact_duplicates"] = {
        "n_dup_groups": len(dup_groups),
        "n_dup_docs": sum(c for _, c in dup_groups),
        "top_groups": [
            {"count": c, "url": fp_examples[fp]["url"],
             "preview": fp_examples[fp]["text"][:240]}
            for fp, c in dup_groups[:TOP_K]
        ],
    }

    # ---- §5.2 near-duplicates (MinHash LSH) -----------------------------
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERMS)
    minhashes: dict[int, MinHash] = {}
    for i, d in enumerate(docs):
        toks = tokenize(normalize_ar(d["text"]))
        if len(toks) < 5:
            continue
        shingles = {" ".join(toks[j:j + 5]) for j in range(len(toks) - 4)}
        m = MinHash(num_perm=MINHASH_PERMS)
        for sh in shingles:
            m.update(sh.encode("utf-8"))
        lsh.insert(i, m)
        minhashes[i] = m

    seen_pairs: set[tuple[int, int]] = set()
    near_dup_clusters: dict[int, set[int]] = {}
    for i, m in minhashes.items():
        for j in lsh.query(m):
            if i == j:
                continue
            a, b = sorted((i, j))
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            near_dup_clusters.setdefault(a, set()).add(a)
            near_dup_clusters[a].add(b)
    res["near_duplicates"] = {
        "threshold": MINHASH_THRESHOLD,
        "num_perm":  MINHASH_PERMS,
        "n_pairs":   len(seen_pairs),
        "n_docs_in_clusters": len({x for c in near_dup_clusters.values() for x in c}),
        "n_clusters": len(near_dup_clusters),
        "example_pairs": [
            {"a_url": docs[a]["url"], "b_url": docs[b]["url"],
             "a_preview": docs[a]["text"][:200], "b_preview": docs[b]["text"][:200]}
            for a, b in list(seen_pairs)[:10]
        ],
    }

    # ---- §6 PII detection -----------------------------------------------
    pii_counts: dict[str, int] = {k: 0 for k in PII_PATTERNS}
    pii_docs:   dict[str, int] = {k: 0 for k in PII_PATTERNS}
    pii_examples: dict[str, list[str]] = {k: [] for k in PII_PATTERNS}
    for d in docs:
        text = d["text"]
        for kind, pat in PII_PATTERNS.items():
            matches = pat.findall(text)
            if matches:
                pii_counts[kind] += len(matches)
                pii_docs[kind]   += 1
                if len(pii_examples[kind]) < 10:
                    pii_examples[kind].append(
                        matches[0] if isinstance(matches[0], str) else " ".join(matches[0])
                    )
    res["pii"] = {
        "counts": pii_counts,
        "docs_containing": pii_docs,
        "examples": pii_examples,
    }

    # ---- §6.2 profanity / offensive blocklist ---------------------------
    offensive_hits = collections.Counter()
    offensive_doc_hits = 0
    for d in docs:
        toks = set(tokenize(normalize_ar(d["text"])))
        hits = toks & OFFENSIVE_WORDS
        if hits:
            offensive_doc_hits += 1
            offensive_hits.update(hits)
    res["offensive"] = {
        "n_docs_with_offensive": offensive_doc_hits,
        "top_terms": offensive_hits.most_common(TOP_K),
    }

    # ---- §7 self-contamination via 50-gram match ------------------------
    # Map every 50-gram to the doc id; collisions = repeated long span.
    ngram_to_doc: dict[str, int] = {}
    collisions = collections.Counter()
    collision_examples: list[dict] = []
    for i, d in enumerate(docs[:CONTAM_MAX_DOCS]):
        toks = tokenize(normalize_ar(d["text"]))
        if len(toks) < CONTAM_NGRAM:
            continue
        seen_local = set()
        for j in range(len(toks) - CONTAM_NGRAM + 1):
            gram = " ".join(toks[j:j + CONTAM_NGRAM])
            if gram in seen_local:
                continue
            seen_local.add(gram)
            if gram in ngram_to_doc and ngram_to_doc[gram] != i:
                collisions[gram] += 1
                if len(collision_examples) < 10:
                    collision_examples.append({
                        "gram_preview": gram[:240],
                        "doc_a_url": docs[ngram_to_doc[gram]]["url"],
                        "doc_b_url": d["url"],
                    })
            else:
                ngram_to_doc[gram] = i
    res["self_contamination"] = {
        "ngram_size": CONTAM_NGRAM,
        "n_unique_long_ngrams": len(ngram_to_doc),
        "n_repeated_long_ngrams": len(collisions),
        "examples": collision_examples,
    }

    # ---- packaged docs list (kept short; full text dropped) -------------
    res["_doc_summaries"] = [{
        "url": d["url"],
        "lang": d["lang"],
        "lang_score": d["lang_score"],
        "n_tokens": d["n_tokens"],
        "n_chars": d["n_chars"],
        "timestamp": d["timestamp"],
    } for d in docs]

    return res


def _dist_stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "max":  float(arr.max()),
        "sum":  float(arr.sum()),
        "samples": arr.tolist() if arr.size <= 5000 else arr[:5000].tolist(),
    }


# ----------------------------------------------------- dashboard generation
PLOT_BG  = "#0e1117"
PAPER_BG = "#0e1117"
FONT_COL = "#e5e7eb"
ACCENT   = "#60a5fa"
ACCENT2  = "#f472b6"
ACCENT3  = "#34d399"


def _layout(title: str, **kw) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=FONT_COL, size=14)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, family="Inter, sans-serif"),
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
        **kw,
    )


def fig_length_hist(stats: dict, title: str, color: str) -> go.Figure:
    samples = stats.get("samples", [])
    fig = go.Figure(go.Histogram(x=samples, nbinsx=50, marker_color=color))
    fig.update_layout(**_layout(title))
    return fig


def fig_topk_bar(items: list[tuple[str, int]], title: str, color: str,
                 horizontal: bool = True, reverse_text: bool = False) -> go.Figure:
    if not items:
        return go.Figure().update_layout(**_layout(title + " (no data)"))
    labels, counts = zip(*items)
    if reverse_text:
        labels = [l[::-1] if regex.search(r"\p{Arabic}", l) else l for l in labels]
    if horizontal:
        fig = go.Figure(go.Bar(
            x=list(counts)[::-1], y=list(labels)[::-1],
            orientation="h", marker_color=color))
    else:
        fig = go.Figure(go.Bar(x=list(labels), y=list(counts), marker_color=color))
    fig.update_layout(**_layout(title), height=max(380, 22 * len(items)))
    return fig


def fig_pie(counts: dict, title: str) -> go.Figure:
    if not counts:
        return go.Figure().update_layout(**_layout(title + " (no data)"))
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()), hole=0.45,
    ))
    fig.update_layout(**_layout(title))
    return fig


def fig_timeline(by_period: dict[str, int], title: str) -> go.Figure:
    if not by_period:
        return go.Figure().update_layout(**_layout(title + " (no data)"))
    fig = go.Figure(go.Bar(
        x=list(by_period.keys()), y=list(by_period.values()),
        marker_color=ACCENT))
    fig.update_layout(**_layout(title))
    return fig


def build_dashboard(res: dict, out_path: Path, input_path: Path) -> None:
    figures: list[tuple[str, go.Figure]] = []
    # length distributions
    figures.append(("Tokens / document",
        fig_length_hist(res["length_stats"]["tokens_per_doc"],
                        "Tokens per document", ACCENT)))
    figures.append(("Chars / document",
        fig_length_hist(res["length_stats"]["chars_per_doc"],
                        "Characters per document", ACCENT2)))
    figures.append(("Bytes / document",
        fig_length_hist(res["length_stats"]["bytes_per_doc"],
                        "Bytes per document", ACCENT3)))
    figures.append(("Alpha ratio",
        fig_length_hist(res["length_stats"]["alpha_ratio"],
                        "Alphabetic-character ratio", ACCENT)))
    figures.append(("Symbol/word ratio",
        fig_length_hist(res["length_stats"]["symbol_per_word"],
                        "Symbols per word (quality signal)", ACCENT2)))
    figures.append(("Mean word length",
        fig_length_hist(res["length_stats"]["mean_word_len"],
                        "Mean word length", ACCENT3)))
    figures.append(("langdetect confidence",
        fig_length_hist(res["lang_score_stats"],
                        "langdetect confidence score", ACCENT)))

    # categorical
    figures.append(("Languages",         fig_pie(res["languages"], "langdetect languages")))
    figures.append(("Content types",     fig_pie(res["content_types"], "Content types")))
    figures.append(("Status",            fig_pie(res["statuses"], "Record status")))

    # URL / domain
    figures.append(("Top domains",  fig_topk_bar(res["top_domains"],  "Top registered domains", ACCENT)))
    figures.append(("Top TLDs",     fig_topk_bar(res["top_tlds"],     "Top top-level domains", ACCENT2)))
    figures.append(("Top URL suffixes",
                    fig_topk_bar(res["top_suffixes"], "Top URL suffixes", ACCENT3)))
    if res["duplicate_urls"]:
        figures.append(("Duplicate URLs",
            fig_topk_bar(res["duplicate_urls"], "Most-repeated URLs", ACCENT)))

    # timeline
    figures.append(("Documents per year",  fig_timeline(res["by_year"],  "Documents per year (CC timestamp)")))
    figures.append(("Documents per month", fig_timeline(res["by_month"], "Documents per month (CC timestamp)")))

    # n-grams (Arabic – reverse text so they render left-to-right correctly
    # in a left-to-right Plotly layout)
    figures.append(("Top unigrams",
        fig_topk_bar(res["top_unigrams"], "Top unigrams (post-normalization, stopwords removed)",
                     ACCENT, reverse_text=True)))
    figures.append(("Top bigrams",
        fig_topk_bar(res["top_bigrams"], "Top bigrams", ACCENT2, reverse_text=True)))
    figures.append(("Top trigrams",
        fig_topk_bar(res["top_trigrams"], "Top trigrams", ACCENT3, reverse_text=True)))

    # word-length distribution
    wl = res["vocab"]["word_length_distribution"]
    figures.append(("Word length distribution",
        fig_topk_bar([(str(k), v) for k, v in wl.items()],
                     "Word length distribution (characters)",
                     ACCENT, horizontal=False)))

    # PII
    figures.append(("PII counts",
        fig_topk_bar(list(res["pii"]["counts"].items()),
                     "Total PII matches by category", ACCENT2, horizontal=False)))
    figures.append(("PII docs",
        fig_topk_bar(list(res["pii"]["docs_containing"].items()),
                     "Documents containing PII by category", ACCENT3, horizontal=False)))

    # offensive
    if res["offensive"]["top_terms"]:
        figures.append(("Offensive terms",
            fig_topk_bar(res["offensive"]["top_terms"],
                         "Offensive / profanity terms found",
                         "#ef4444", reverse_text=True)))

    # ---- assemble HTML --------------------------------------------------
    summary_cards = [
        ("Documents",        f"{res['totals']['n_documents']:,}"),
        ("Tokens",           f"{res['totals']['n_tokens']:,}"),
        ("Characters",       f"{res['totals']['n_chars']:,}"),
        ("Bytes",            f"{res['totals']['n_bytes']:,}"),
        ("Unique URLs",      f"{res['totals']['n_unique_urls']:,}"),
        ("Vocabulary types", f"{res['vocab']['n_types']:,}"),
        ("Exact-dup groups", f"{res['exact_duplicates']['n_dup_groups']:,}"),
        ("Near-dup clusters",f"{res['near_duplicates']['n_clusters']:,}"),
        ("Near-dup pairs",   f"{res['near_duplicates']['n_pairs']:,}"),
        ("Repeated 50-grams",f"{res['self_contamination']['n_repeated_long_ngrams']:,}"),
        ("PII e-mails",      f"{res['pii']['counts']['email']:,}"),
        ("PII phones",       f"{res['pii']['counts']['phone']:,}"),
        ("Offensive docs",   f"{res['offensive']['n_docs_with_offensive']:,}"),
    ]

    fig_html_blocks: list[str] = []
    for _, fig in figures:
        fig_html_blocks.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

    # tables
    dup_rows = "".join(
        f"<tr><td>{g['count']}</td>"
        f"<td><a href='{html.escape(g['url'] or '')}' target='_blank'>"
        f"{html.escape((g['url'] or '')[:80])}</a></td>"
        f"<td><div class='rtl'>{html.escape(g['preview'])}</div></td></tr>"
        for g in res["exact_duplicates"]["top_groups"][:20]
    )
    near_rows = "".join(
        f"<tr>"
        f"<td><a href='{html.escape(p['a_url'] or '')}' target='_blank'>"
        f"{html.escape((p['a_url'] or '')[:60])}</a></td>"
        f"<td><a href='{html.escape(p['b_url'] or '')}' target='_blank'>"
        f"{html.escape((p['b_url'] or '')[:60])}</a></td>"
        f"<td><div class='rtl'>{html.escape(p['a_preview'])}</div></td>"
        f"<td><div class='rtl'>{html.escape(p['b_preview'])}</div></td>"
        f"</tr>"
        for p in res["near_duplicates"]["example_pairs"]
    )
    pii_examples_rows = "".join(
        f"<tr><td>{k}</td><td>{res['pii']['counts'][k]}</td>"
        f"<td>{res['pii']['docs_containing'][k]}</td>"
        f"<td>{html.escape(', '.join(v[:5]))}</td></tr>"
        for k, v in res["pii"]["examples"].items()
    )
    contam_rows = "".join(
        f"<tr>"
        f"<td><a href='{html.escape(c['doc_a_url'] or '')}' target='_blank'>doc A</a></td>"
        f"<td><a href='{html.escape(c['doc_b_url'] or '')}' target='_blank'>doc B</a></td>"
        f"<td><div class='rtl'>{html.escape(c['gram_preview'])}</div></td>"
        f"</tr>"
        for c in res["self_contamination"]["examples"]
    )

    cards_html = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(k)}</div>"
        f"<div class='card-value'>{html.escape(v)}</div></div>"
        for k, v in summary_cards
    )
    figs_html = "".join(
        f"<section class='panel'><h3>{html.escape(title)}</h3>{block}</section>"
        for (title, _), block in zip(figures, fig_html_blocks)
    )

    html_doc = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>WIMBD — Arabic CC sample dashboard</title>
<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>
<style>
  body {{ background:#0b0f17; color:#e5e7eb; font-family: Inter,system-ui,sans-serif;
         margin: 0; padding: 24px; }}
  h1 {{ margin: 0 0 4px 0; font-weight: 600; }}
  .sub {{ color:#94a3b8; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr));
            gap: 12px; margin-bottom: 28px; }}
  .card {{ background:#111827; padding:14px 16px; border:1px solid #1f2937;
           border-radius:10px; }}
  .card-label {{ color:#94a3b8; font-size: 12px; text-transform: uppercase;
                 letter-spacing: 0.05em; }}
  .card-value {{ font-size: 22px; font-weight:600; margin-top: 4px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(540px,1fr));
           gap: 18px; }}
  .panel {{ background:#0e1117; border:1px solid #1f2937; border-radius:10px;
            padding: 8px 8px 4px 8px; }}
  .panel h3 {{ margin:6px 12px 0 12px; font-size:14px; color:#94a3b8;
               font-weight: 500; }}
  details {{ background:#0e1117; border:1px solid #1f2937; border-radius:10px;
             padding: 12px 16px; margin-top: 18px; }}
  details > summary {{ cursor:pointer; font-weight:600; color:#cbd5e1; }}
  table {{ border-collapse: collapse; width:100%; margin-top: 10px;
           font-size: 13px; }}
  td, th {{ border-bottom:1px solid #1f2937; padding:6px 8px;
            vertical-align: top; }}
  th {{ color:#94a3b8; text-align:left; font-weight: 500; }}
  a {{ color:#60a5fa; text-decoration:none; }}
  .rtl {{ direction: rtl; text-align: right; }}
  .meta {{ font-size:12px; color:#64748b; margin-top: 32px; }}
</style>
</head>
<body>
  <h1>WIMBD — Arabic Common Crawl sample</h1>
  <div class='sub'>{html.escape(str(input_path))} ·
                   analyses adapted from <em>What's In My Big Data?</em>
                   (Elazar et&nbsp;al., 2023, arXiv:2310.20707)</div>
  <div class='cards'>{cards_html}</div>

  <h2>Distributions &amp; counts</h2>
  <div class='grid'>{figs_html}</div>

  <details open><summary>Exact-duplicate document groups (top 20)</summary>
    <table><thead><tr><th>count</th><th>example URL</th><th>preview</th></tr></thead>
    <tbody>{dup_rows or '<tr><td colspan=3>None</td></tr>'}</tbody></table>
  </details>

  <details open><summary>Near-duplicate pairs (MinHash &gt;= {MINHASH_THRESHOLD})</summary>
    <table><thead><tr><th>doc A URL</th><th>doc B URL</th>
                      <th>A preview</th><th>B preview</th></tr></thead>
    <tbody>{near_rows or '<tr><td colspan=4>None</td></tr>'}</tbody></table>
  </details>

  <details open><summary>PII examples</summary>
    <table><thead><tr><th>category</th><th>total matches</th>
                      <th># docs</th><th>example values</th></tr></thead>
    <tbody>{pii_examples_rows}</tbody></table>
  </details>

  <details><summary>Self-contamination: repeated {CONTAM_NGRAM}-grams</summary>
    <table><thead><tr><th>doc A</th><th>doc B</th>
                      <th>repeated span</th></tr></thead>
    <tbody>{contam_rows or '<tr><td colspan=3>None</td></tr>'}</tbody></table>
  </details>

  <div class='meta'>Generated by wimbd_arabic.py</div>
</body></html>"""

    out_path.write_text(html_doc, encoding="utf-8")


# ----------------------------------------------------- markdown text report
def write_report(res: dict, out_path: Path, input_path: Path) -> None:
    t = res["totals"]
    ls = res["length_stats"]
    lines = [
        f"# WIMBD report — {input_path.name}",
        "",
        f"- documents: **{t['n_documents']:,}**",
        f"- tokens: **{t['n_tokens']:,}**  /  characters: **{t['n_chars']:,}**  /  bytes: **{t['n_bytes']:,}**",
        f"- unique URLs: **{t['n_unique_urls']:,}** "
        f"(normalized: {t['n_unique_normalized_urls']:,})",
        f"- vocabulary types: **{res['vocab']['n_types']:,}**",
        "",
        "## length stats",
        f"- tokens/doc: mean={ls['tokens_per_doc']['mean']:.1f}  "
        f"p50={ls['tokens_per_doc']['p50']:.0f}  "
        f"p95={ls['tokens_per_doc']['p95']:.0f}  "
        f"max={ls['tokens_per_doc']['max']:.0f}",
        f"- chars/doc:  mean={ls['chars_per_doc']['mean']:.1f}  "
        f"p50={ls['chars_per_doc']['p50']:.0f}  "
        f"p95={ls['chars_per_doc']['p95']:.0f}",
        f"- alpha ratio mean: {ls['alpha_ratio']['mean']:.3f}",
        f"- symbol/word ratio mean: {ls['symbol_per_word']['mean']:.3f}",
        "",
        "## duplication",
        f"- exact-duplicate document groups: {res['exact_duplicates']['n_dup_groups']:,}",
        f"- documents in those groups: {res['exact_duplicates']['n_dup_docs']:,}",
        f"- near-duplicate clusters (MinHash >= {res['near_duplicates']['threshold']}): "
        f"{res['near_duplicates']['n_clusters']:,}",
        f"- near-duplicate doc-pairs: {res['near_duplicates']['n_pairs']:,}",
        f"- repeated {res['self_contamination']['ngram_size']}-grams: "
        f"{res['self_contamination']['n_repeated_long_ngrams']:,}",
        "",
        "## PII",
        *[f"- {k}: {v:,} matches across "
          f"{res['pii']['docs_containing'][k]:,} docs" for k, v in res["pii"]["counts"].items()],
        "",
        "## offensive",
        f"- documents with any term from the small Arabic blocklist: "
        f"{res['offensive']['n_docs_with_offensive']:,}",
        "",
        "## top URL hosts",
        *[f"- {d}: {c}" for d, c in res["top_domains"][:10]],
        "",
        "## top unigrams",
        *[f"- {w}: {c}" for w, c in res["top_unigrams"][:15]],
        "",
        "## top bigrams",
        *[f"- {w}: {c}" for w, c in res["top_bigrams"][:15]],
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------- entry
def main() -> None:
    t0 = time.time()
    print(f"[wimbd] reading: {INPUT_PATH}")
    records = list(stream_records(INPUT_PATH))
    print(f"[wimbd] loaded {len(records):,} records "
          f"in {time.time() - t0:.1f}s")

    print("[wimbd] running analyses ...")
    res = analyze(records)

    json_out = OUT_DIR / "wimbd_results.json"
    html_out = OUT_DIR / "wimbd_dashboard.html"
    md_out   = OUT_DIR / "wimbd_report.md"

    # drop big sample arrays from JSON to keep file small
    res_for_json = json.loads(json.dumps(res, default=str))
    for k in ("tokens_per_doc", "chars_per_doc", "bytes_per_doc",
              "alpha_ratio", "symbol_per_word", "mean_word_len"):
        res_for_json["length_stats"][k].pop("samples", None)
    res_for_json["lang_score_stats"].pop("samples", None)
    res_for_json.pop("_doc_summaries", None)

    json_out.write_text(json.dumps(res_for_json, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    build_dashboard(res, html_out, INPUT_PATH)
    write_report(res, md_out, INPUT_PATH)

    print(f"[wimbd] wrote {json_out}")
    print(f"[wimbd] wrote {html_out}")
    print(f"[wimbd] wrote {md_out}")
    print(f"[wimbd] total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
