"""
Unit test suite for wimbd_arabic.py
====================================
Corpus: state_output_sample1000.jsonl  (1,000 real Arabic CC records)

Ground truth (measured on this corpus):
  n_documents          : 1000
  n_tokens             : 1,028,842
  n_bytes              : 10,097,974
  n_unique_urls        : 1000
  statuses             : PASSED×512, FAILED F5×281, FAILED F6×117,
                         FAILED F1×54, FAILED F2×36
  languages            : ar×1000
  content_types        : text/plain×1000
  by_year              : {2017: 1000}
  exact_dup_groups     : 3  (10 docs involved)
  near_dup_pairs       : 109  (47 clusters)
  pii email            : 76 matches in 51 docs
  pii ipv4             : 17 matches in 13 docs
  offensive_docs       : 30
  vocab_types          : 103,040
  lang_score_mean      : 0.9506
  self_contam_repeated : 70,778

Run all tests:
    pytest test_wimbd_arabic.py -v -s -W ignore::DeprecationWarning
"""
from __future__ import annotations

import collections
import hashlib
import re
import sys
import time
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import wimbd_arabic without triggering its mkdir / file side-effects
# ---------------------------------------------------------------------------
with mock.patch("pathlib.Path.mkdir"):
    sys.path.insert(0, str(Path(__file__).parent))
    import wimbd_arabic as W

CORPUS = Path(__file__).parent / "state_output_sample1000.jsonl"


# ===========================================================================
# SESSION-SCOPED FIXTURES  (corpus loaded + analyzed exactly once)
# ===========================================================================

@pytest.fixture(scope="session")
def records():
    assert CORPUS.exists(), f"Corpus not found: {CORPUS}"
    return list(W.stream_records(CORPUS))


@pytest.fixture(scope="session")
def analysis(records):
    return W.analyze(records)


# ===========================================================================
# 1. Arabic text normalisation
# ===========================================================================

class TestNormalizeArabic:

    def test_removes_tashkeel(self):
        for mark in ["ُ", "َ", "ِ", "ً", "ٌ", "ٍ", "ّ"]:
            text, result = f"ك{mark}تاب", W.normalize_ar(f"ك{mark}تاب")
            print(f"\n  IN: {text!r}  →  OUT: {result!r}")
            assert mark not in result

    def test_unifies_alef_hamza_above(self):
        a, b = "أحمد", "احمد"
        print(f"\n  IN: {a!r}  →  OUT: {W.normalize_ar(a)!r}")
        print(f"  IN: {b!r}  →  OUT: {W.normalize_ar(b)!r}")
        assert W.normalize_ar(a) == W.normalize_ar(b)

    def test_unifies_alef_hamza_below(self):
        a, b = "إسلام", "اسلام"
        print(f"\n  IN: {a!r}  →  OUT: {W.normalize_ar(a)!r}")
        print(f"  IN: {b!r}  →  OUT: {W.normalize_ar(b)!r}")
        assert W.normalize_ar(a) == W.normalize_ar(b)

    def test_unifies_alef_madda(self):
        a, b = "آمين", "امين"
        print(f"\n  IN: {a!r}  →  OUT: {W.normalize_ar(a)!r}")
        print(f"  IN: {b!r}  →  OUT: {W.normalize_ar(b)!r}")
        assert W.normalize_ar(a) == W.normalize_ar(b)

    def test_unifies_alef_maqsura(self):
        a, b = "مبنى", "مبني"
        print(f"\n  IN: {a!r}  →  OUT: {W.normalize_ar(a)!r}")
        print(f"  IN: {b!r}  →  OUT: {W.normalize_ar(b)!r}")
        assert W.normalize_ar(a) == W.normalize_ar(b)

    def test_unifies_ta_marbuta(self):
        a, b = "مدرسة", "مدرسه"
        print(f"\n  IN: {a!r}  →  OUT: {W.normalize_ar(a)!r}")
        print(f"  IN: {b!r}  →  OUT: {W.normalize_ar(b)!r}")
        assert W.normalize_ar(a) == W.normalize_ar(b)

    def test_removes_tatweel(self):
        text, result = "جمـيل", W.normalize_ar("جمـيل")
        print(f"\n  IN: {text!r}  →  OUT: {result!r}")
        assert "ـ" not in result

    def test_empty_string(self):
        result = W.normalize_ar("")
        print(f"\n  IN: ''  →  OUT: {result!r}")
        assert result == ""

    def test_latin_chars_unaffected(self):
        text, result = "Python 3.12", W.normalize_ar("Python 3.12")
        print(f"\n  IN: {text!r}  →  OUT: {result!r}")
        assert "Python" in result


# ===========================================================================
# 2. Tokeniser
# ===========================================================================

class TestTokenize:

    def test_arabic_words(self):
        text, tokens = "الذكاء الاصطناعي", W.tokenize("الذكاء الاصطناعي")
        print(f"\n  IN: {text!r}  →  OUT: {tokens}")
        assert "الذكاء" in tokens and "الاصطناعي" in tokens

    def test_mixed_arabic_latin(self):
        text, tokens = "تعلم Python بسهولة", W.tokenize("تعلم Python بسهولة")
        print(f"\n  IN: {text!r}  →  OUT: {tokens}")
        assert "Python" in tokens and "تعلم" in tokens

    def test_numbers_included(self):
        text, tokens = "عام 2017", W.tokenize("عام 2017")
        print(f"\n  IN: {text!r}  →  OUT: {tokens}")
        assert "2017" in tokens

    def test_punctuation_excluded(self):
        text, tokens = "مرحبا، كيف حالك؟", W.tokenize("مرحبا، كيف حالك؟")
        print(f"\n  IN: {text!r}  →  OUT: {tokens}")
        assert "،" not in tokens and "؟" not in tokens

    def test_empty_string(self):
        result = W.tokenize("")
        print(f"\n  IN: ''  →  OUT: {result}")
        assert result == []

    def test_real_doc_tokenizes_to_nonzero(self, records):
        text = records[0]["record"].get("text", "")
        tokens = W.tokenize(text)
        print(f"\n  first doc preview: {text[:60]!r}")
        print(f"  token count: {len(tokens)}, first 5: {tokens[:5]}")
        assert len(tokens) > 0


# ===========================================================================
# 3. IO helpers
# ===========================================================================

class TestStreamRecords:

    def test_loads_1000_records(self, records):
        assert len(records) == 1000

    def test_every_record_has_record_and_status(self, records):
        for r in records:
            assert "record" in r and "status" in r

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text(
            '{"record":{},"status":"PASSED"}\n\n{"record":{},"status":"PASSED"}\n',
            encoding="utf-8",
        )
        assert len(list(W.stream_records(f))) == 2

    def test_skips_malformed_json(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text(
            '{"record":{},"status":"PASSED"}\nBAD\n{"record":{},"status":"PASSED"}\n',
            encoding="utf-8",
        )
        assert len(list(W.stream_records(f))) == 2

    def test_empty_file(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text("", encoding="utf-8")
        assert list(W.stream_records(f)) == []

    def test_arabic_unicode_preserved(self, tmp_path):
        f = tmp_path / "t.jsonl"
        f.write_text('{"record":{"text":"مرحبا"},"status":"PASSED"}\n', encoding="utf-8")
        assert list(W.stream_records(f))[0]["record"]["text"] == "مرحبا"


class TestFingerprint:

    def test_deterministic(self):
        t = "الذكاء الاصطناعي"
        r1, r2 = W.fingerprint(t), W.fingerprint(t)
        print(f"\n  IN: {t!r}  →  OUT: {r1!r}  (called twice, same result: {r1 == r2})")
        assert r1 == r2

    def test_different_texts_differ(self):
        a, b = "نص أول", "نص ثانٍ"
        fa, fb = W.fingerprint(a), W.fingerprint(b)
        print(f"\n  IN: {a!r}  →  OUT: {fa!r}")
        print(f"  IN: {b!r}  →  OUT: {fb!r}")
        assert fa != fb

    def test_returns_32_char_hex(self):
        text, result = "test", W.fingerprint("test")
        print(f"\n  IN: {text!r}  →  OUT: {result!r}  (len={len(result)})")
        assert re.fullmatch(r"[0-9a-f]{32}", result)

    def test_matches_hashlib_md5(self):
        text = "مرحبا"
        result = W.fingerprint(text)
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        print(f"\n  IN: {text!r}  →  fingerprint: {result!r}  ==  hashlib.md5: {expected!r}")
        assert result == expected

    def test_whitespace_sensitive(self):
        a, b = "a b", "a  b"
        fa, fb = W.fingerprint(a), W.fingerprint(b)
        print(f"\n  IN: {a!r}  →  OUT: {fa!r}")
        print(f"  IN: {b!r}  →  OUT: {fb!r}")
        assert fa != fb


# ===========================================================================
# 4. PII detection
# ===========================================================================

class TestPIIPatterns:

    def _find(self, kind, text):
        return W.PII_PATTERNS[kind].findall(text)

    def test_email_basic(self):
        text, matches = "user@example.com", self._find("email", "user@example.com")
        print(f"\n  IN: {text!r}  →  email matches: {matches}")
        assert matches

    def test_email_real_corpus_example(self):
        text, matches = "ahmed_club2000@yahoo.com", self._find("email", "ahmed_club2000@yahoo.com")
        print(f"\n  IN: {text!r}  →  email matches: {matches}")
        assert matches

    def test_email_in_arabic_sentence(self):
        text, matches = "راسلنا على info@kds.ae اليوم", self._find("email", "راسلنا على info@kds.ae اليوم")
        print(f"\n  IN: {text!r}  →  email matches: {matches}")
        assert matches

    def test_email_no_match_incomplete(self):
        for text in ["user@", "@example.com"]:
            matches = self._find("email", text)
            print(f"\n  IN: {text!r}  →  email matches: {matches}  (expected none)")
        assert not self._find("email", "user@") and not self._find("email", "@example.com")

    def test_ipv4_standard(self):
        text, matches = "192.168.1.1", self._find("ipv4", "192.168.1.1")
        print(f"\n  IN: {text!r}  →  ipv4 matches: {matches}")
        assert matches

    def test_ipv4_real_corpus_example(self):
        text, matches = "217.218.48.21", self._find("ipv4", "217.218.48.21")
        print(f"\n  IN: {text!r}  →  ipv4 matches: {matches}")
        assert matches

    def test_ipv4_no_match_three_octets(self):
        text, matches = "192.168.1", self._find("ipv4", "192.168.1")
        print(f"\n  IN: {text!r}  →  ipv4 matches: {matches}  (expected none)")
        assert not matches

    def test_url_https(self):
        text, matches = "https://example.com/path", self._find("url", "https://example.com/path")
        print(f"\n  IN: {text!r}  →  url matches: {matches}")
        assert matches

    def test_url_no_match_without_scheme(self):
        text, matches = "example.com/page", self._find("url", "example.com/page")
        print(f"\n  IN: {text!r}  →  url matches: {matches}  (expected none)")
        assert not matches

    def test_phone_international(self):
        text, matches = "+966-555-1234", self._find("phone", "+966-555-1234")
        print(f"\n  IN: {text!r}  →  phone matches: {matches}")
        assert matches

    def test_phone_local(self):
        text, matches = "0501234567", self._find("phone", "0501234567")
        print(f"\n  IN: {text!r}  →  phone matches: {matches}")
        assert matches

    def test_phone_arabic_indic_digits(self):
        text, matches = "٠٥٠١٢٣٤٥٦٧", self._find("phone", "٠٥٠١٢٣٤٥٦٧")
        print(f"\n  IN: {text!r}  →  phone matches: {matches}")
        assert matches


class TestPIIFromRealCorpus:

    def test_email_count(self, analysis):
        assert analysis["pii"]["counts"]["email"] == 76

    def test_email_docs(self, analysis):
        assert analysis["pii"]["docs_containing"]["email"] == 51

    def test_ipv4_count(self, analysis):
        assert analysis["pii"]["counts"]["ipv4"] == 17

    def test_ipv4_docs(self, analysis):
        assert analysis["pii"]["docs_containing"]["ipv4"] == 13

    def test_examples_captured(self, analysis):
        for cat in ("email", "ipv4"):
            assert len(analysis["pii"]["examples"][cat]) >= 1

    def test_all_categories_present(self, analysis):
        for cat in ("email", "url", "ipv4", "phone", "card_like"):
            assert cat in analysis["pii"]["counts"]


# ===========================================================================
# 5. Offensive content
# ===========================================================================

class TestOffensiveDetection:

    def _run(self, texts):
        hits, n_docs = collections.Counter(), 0
        for text in texts:
            matched = set(W.tokenize(W.normalize_ar(text))) & W.OFFENSIVE_WORDS
            if matched:
                n_docs += 1
                hits.update(matched)
        return n_docs, hits

    def test_known_word_detected(self):
        text = "هذا الرجل مثل كلب"
        n, hits = self._run([text])
        print(f"\n  IN: {text!r}  →  hits: {dict(hits)}, docs: {n}")
        assert n == 1

    def test_clean_text_zero_hits(self):
        text = "الذكاء الاصطناعي مفيد للغاية"
        n, hits = self._run([text])
        print(f"\n  IN: {text!r}  →  hits: {dict(hits)}, docs: {n}")
        assert n == 0

    def test_normalization_applied_before_match(self):
        text = "هذا كَلْبٌ"
        n, hits = self._run([text])
        print(f"\n  IN: {text!r}  →  hits: {dict(hits)}, docs: {n}")
        assert n == 1

    def test_space_separated_multiple_words(self):
        text = "كلب حمار"
        n, terms = self._run([text])
        print(f"\n  IN: {text!r}  →  hits: {dict(terms)}, docs: {n}")
        assert n == 1 and len(terms) >= 2

    def test_waw_prefix_not_matched(self):
        # "وكلب" is one token — documented gap vs WIMBD paper
        text = "وكلب"
        n, hits = self._run([text])
        print(f"\n  IN: {text!r}  →  hits: {dict(hits)}, docs: {n}  (waw prefix not split)")
        assert n == 0

    def test_real_corpus_offensive_docs(self, analysis):
        n = analysis["offensive"]["n_docs_with_offensive"]
        top = analysis["offensive"]["top_terms"][:5]
        print(f"\n  offensive docs in corpus: {n}")
        print(f"  top terms: {top}")
        assert n == 30


# ===========================================================================
# 6. Exact duplicate detection
# ===========================================================================

class TestExactDuplicates:

    def test_three_dup_groups(self, analysis):
        n = analysis["exact_duplicates"]["n_dup_groups"]
        print(f"\n  exact duplicate groups found: {n}")
        for g in analysis["exact_duplicates"]["top_groups"]:
            print(f"    count={g['count']}  url={g['url']}  preview={g['preview'][:60]!r}")
        assert n == 3

    def test_ten_docs_in_dup_groups(self, analysis):
        n = analysis["exact_duplicates"]["n_dup_docs"]
        print(f"\n  total docs involved in exact duplicates: {n}")
        assert n == 10

    def test_all_top_groups_count_gt_1(self, analysis):
        for g in analysis["exact_duplicates"]["top_groups"]:
            assert g["count"] > 1

    def test_group_has_expected_keys(self, analysis):
        for g in analysis["exact_duplicates"]["top_groups"]:
            assert {"count", "url", "preview"} <= g.keys()

    def test_no_dups_in_clean_corpus(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": f"نص فريد {i}",
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(5)
        ]
        result = W.analyze(recs)
        n = result["exact_duplicates"]["n_dup_groups"]
        print(f"\n  clean corpus (5 unique docs)  →  dup groups: {n}")
        assert n == 0

    def test_fingerprint_differs_for_near_dups(self):
        a, b = "نص أ", "نص ب"
        fa, fb = W.fingerprint(a), W.fingerprint(b)
        print(f"\n  IN: {a!r}  →  {fa!r}")
        print(f"  IN: {b!r}  →  {fb!r}  (different ✓)")
        assert fa != fb


# ===========================================================================
# 7. Near-duplicate detection (MinHash LSH)
# ===========================================================================

class TestNearDuplicates:

    def test_pair_count(self, analysis):
        n = analysis["near_duplicates"]["n_pairs"]
        print(f"\n  near-duplicate pairs found: {n}  (threshold={analysis['near_duplicates']['threshold']})")
        for p in analysis["near_duplicates"]["example_pairs"][:3]:
            print(f"    A: {p['a_url']}")
            print(f"    B: {p['b_url']}")
            print(f"    A preview: {p['a_preview'][:60]!r}")
            print(f"    B preview: {p['b_preview'][:60]!r}")
            print()
        assert n == 109

    def test_cluster_count(self, analysis):
        n = analysis["near_duplicates"]["n_clusters"]
        print(f"\n  near-duplicate clusters: {n}")
        assert n == 47

    def test_threshold_and_perms(self, analysis):
        t = analysis["near_duplicates"]["threshold"]
        p = analysis["near_duplicates"]["num_perm"]
        print(f"\n  MinHash threshold={t}, num_perm={p}")
        assert t == 0.8 and p == 128

    def test_short_docs_not_indexed(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": "كلمة",
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(3)
        ]
        result = W.analyze(recs)
        n = result["near_duplicates"]["n_pairs"]
        print(f"\n  3 single-token docs  →  near-dup pairs: {n}  (short docs skipped ✓)")
        assert n == 0

    def test_identical_long_docs_pair(self):
        long = "الذكاء الاصطناعي يعتمد على التعلم الآلي " * 10
        recs = [
            {"record": {"url": f"https://site{i}.com", "text": long,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(2)
        ]
        result = W.analyze(recs)
        n = result["near_duplicates"]["n_pairs"]
        print(f"\n  2 identical long docs  →  near-dup pairs: {n}  (detected ✓)")
        assert n >= 1

    def test_unrelated_docs_no_pair(self):
        recs = [
            {"record": {"url": "https://a.com",
                        "text": "الذكاء الاصطناعي تعلم آلة بيانات خوارزميات " * 8,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"},
            {"record": {"url": "https://b.com",
                        "text": "كرة قدم رياضة ملعب هدف بطولة دوري فريق لاعب " * 8,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"},
        ]
        result = W.analyze(recs)
        n = result["near_duplicates"]["n_pairs"]
        print(f"\n  2 unrelated docs (AI vs football)  →  near-dup pairs: {n}  (correctly 0 ✓)")
        assert n == 0


# ===========================================================================
# 8. N-gram analysis
# ===========================================================================

class TestNgramAnalysis:

    def test_unigrams_populated(self, analysis):
        top = analysis["top_unigrams"][:10]
        print(f"\n  top unigrams (stopwords removed):")
        for word, count in top:
            print(f"    {word}: {count}")
        assert len(top) > 0

    def test_unigrams_are_word_int_tuples(self, analysis):
        for word, count in analysis["top_unigrams"]:
            assert isinstance(word, str) and isinstance(count, int) and count > 0

    def test_no_stopwords_in_unigrams(self, analysis):
        for word, _ in analysis["top_unigrams"]:
            assert word not in W.ARABIC_STOPWORDS

    def test_bigrams_are_two_tokens(self, analysis):
        top = analysis["top_bigrams"][:5]
        print(f"\n  top bigrams:")
        for bigram, count in top:
            print(f"    {bigram!r}: {count}")
            assert len(bigram.split(" ")) == 2

    def test_trigrams_are_three_tokens(self, analysis):
        top = analysis["top_trigrams"][:5]
        print(f"\n  top trigrams:")
        for trigram, count in top:
            print(f"    {trigram!r}: {count}")
            assert len(trigram.split(" ")) == 3

    def test_vocab_size(self, analysis):
        n = analysis["vocab"]["n_types"]
        print(f"\n  vocabulary size (unique tokens after normalization + stopword removal): {n:,}")
        assert n == 102115

    def test_normalization_merges_alef_variants(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": t,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i, t in enumerate(["أحمد يدرس الهندسة", "احمد يدرس الهندسة"])
        ]
        unigrams = dict(W.analyze(recs)["top_unigrams"])
        count = unigrams.get("احمد", 0)
        print(f"\n  'أحمد' and 'احمد' both normalize to 'احمد'  →  count: {count}")
        assert count == 2


# ===========================================================================
# 9. URL / domain analysis
# ===========================================================================

class TestURLDomainAnalysis:

    def test_all_urls_unique(self, analysis):
        assert analysis["totals"]["n_unique_urls"] == 1000

    def test_com_is_top_tld(self, analysis):
        tlds = dict(analysis["top_tlds"])
        assert tlds["com"] == 619

    def test_net_tld_present(self, analysis):
        assert "net" in dict(analysis["top_tlds"])

    def test_top_domains_populated(self, analysis):
        assert len(analysis["top_domains"]) > 0

    def test_null_url_handled(self):
        recs = [{"record": {"url": None, "text": "نص",
                             "normalized_url": None, "timestamp": None,
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        assert W.analyze(recs)["totals"]["n_unique_urls"] == 0


# ===========================================================================
# 10. Date distribution
# ===========================================================================

class TestDateDistribution:

    def test_single_year_2017(self, analysis):
        assert list(analysis["by_year"].keys()) == ["2017"]

    def test_all_1000_docs_in_2017(self, analysis):
        assert analysis["by_year"]["2017"] == 1000

    def test_month_keys_are_yyyy_mm(self, analysis):
        for key in analysis["by_month"]:
            assert re.fullmatch(r"\d{4}-\d{2}", key)

    def test_month_counts_sum_to_1000(self, analysis):
        assert sum(analysis["by_month"].values()) == 1000

    def test_null_timestamp_skipped(self):
        recs = [{"record": {"url": "https://x.com", "text": "نص",
                             "normalized_url": None, "timestamp": None,
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        assert W.analyze(recs)["by_year"] == {}

    def test_invalid_timestamp_skipped(self):
        recs = [{"record": {"url": "https://x.com", "text": "نص",
                             "normalized_url": None, "timestamp": "not-a-date",
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        assert W.analyze(recs)["by_year"] == {}


# ===========================================================================
# 11. Language distribution
# ===========================================================================

class TestLanguageDistribution:

    def test_all_docs_arabic(self, analysis):
        assert analysis["languages"] == {"ar": 1000}

    def test_content_types_all_plain_text(self, analysis):
        assert analysis["content_types"] == {"text/plain": 1000}

    def test_lang_score_mean(self, analysis):
        assert abs(analysis["lang_score_stats"]["mean"] - 0.9506) < 0.001

    def test_lang_score_min_above_zero(self, analysis):
        assert analysis["lang_score_stats"]["min"] > 0.0

    def test_statuses(self, analysis):
        s = analysis["statuses"]
        assert s["PASSED"] == 512
        assert s["FAILED F5"] == 281
        assert s["FAILED F6"] == 117
        assert s["FAILED F1"] == 54
        assert s["FAILED F2"] == 36

    def test_null_langdetect_handled(self):
        recs = [{"record": {"url": "https://x.com", "text": "نص",
                             "normalized_url": None, "timestamp": None,
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        # Must not raise
        W.analyze(recs)


# ===========================================================================
# 12. Dataset statistics & quality signals
# ===========================================================================

class TestDatasetStatistics:

    def test_document_count(self, analysis):
        assert analysis["totals"]["n_documents"] == 1000

    def test_token_count(self, analysis):
        assert analysis["totals"]["n_tokens"] == 1028842

    def test_byte_count(self, analysis):
        assert analysis["totals"]["n_bytes"] == 10097974

    def test_alpha_ratio_in_unit_interval(self, analysis):
        mean = analysis["length_stats"]["alpha_ratio"]["mean"]
        assert 0.0 <= mean <= 1.0

    def test_symbol_word_ratio_non_negative(self, analysis):
        assert analysis["length_stats"]["symbol_per_word"]["mean"] >= 0.0

    def test_empty_doc_zero_tokens(self):
        recs = [{"record": {"url": "https://x.com", "text": "",
                             "normalized_url": None, "timestamp": None,
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        assert W.analyze(recs)["totals"]["n_tokens"] == 0

    def test_mean_word_length_known_input(self):
        # "مرحبا"=5, "عالم"=4 → mean = 4.5
        recs = [{"record": {"url": "https://x.com", "text": "مرحبا عالم",
                             "normalized_url": None, "timestamp": None,
                             "content_type": None, "langdetect": None},
                 "status": "PASSED"}]
        mwl = W.analyze(recs)["length_stats"]["mean_word_len"]["mean"]
        assert abs(mwl - 4.5) < 0.1

    def test_null_fields_do_not_crash(self):
        recs = [{"record": {"url": None, "normalized_url": None, "text": None,
                             "timestamp": None, "content_length": None,
                             "content_type": None, "language": None,
                             "langdetect": None, "sinan_id": None},
                 "status": None}]
        W.analyze(recs)


class TestDistStats:

    def test_empty_returns_n_zero(self):
        assert W._dist_stats([]) == {"n": 0}

    def test_single_value(self):
        r = W._dist_stats([7.0])
        assert r["n"] == 1 and r["mean"] == r["min"] == r["max"] == 7.0

    def test_percentiles_monotone(self):
        r = W._dist_stats(list(range(100)))
        assert r["min"] <= r["p25"] <= r["p50"] <= r["p75"] <= r["p95"] <= r["p99"] <= r["max"]

    def test_sum_correct(self):
        assert abs(W._dist_stats([1.0, 2.0, 3.0, 4.0])["sum"] - 10.0) < 1e-9

    def test_samples_capped_at_5000(self):
        assert len(W._dist_stats(list(range(10_000)))["samples"]) == 5000

    def test_std_zero_for_uniform(self):
        assert W._dist_stats([5.0] * 10)["std"] == 0.0


# ===========================================================================
# 13. Self-contamination
# ===========================================================================

class TestSelfContamination:

    def test_repeated_ngram_count(self, analysis):
        n = analysis["self_contamination"]["n_repeated_long_ngrams"]
        print(f"\n  repeated 50-grams found in corpus: {n:,}")
        for ex in analysis["self_contamination"]["examples"][:2]:
            print(f"    doc A: {ex['doc_a_url']}")
            print(f"    doc B: {ex['doc_b_url']}")
            print(f"    repeated span: {ex['gram_preview'][:80]!r}")
            print()
        assert n == 70778

    def test_ngram_size_is_50(self, analysis):
        n = analysis["self_contamination"]["ngram_size"]
        print(f"\n  contamination n-gram size: {n}")
        assert n == 50

    def test_examples_have_required_keys(self, analysis):
        for ex in analysis["self_contamination"]["examples"]:
            assert {"gram_preview", "doc_a_url", "doc_b_url"} <= ex.keys()

    def test_no_contamination_in_short_corpus(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": f"نص قصير {i}",
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(5)
        ]
        result = W.analyze(recs)
        n = result["self_contamination"]["n_repeated_long_ngrams"]
        print(f"\n  5 short unique docs  →  repeated 50-grams: {n}  (none expected ✓)")
        assert n == 0

    def test_contamination_detected_with_shared_long_span(self):
        span = " ".join([f"كلمة{i}" for i in range(60)])
        recs = [
            {"record": {"url": f"https://doc{i}.com",
                        "text": span + f" نهاية {i}",
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(2)
        ]
        result = W.analyze(recs)
        n = result["self_contamination"]["n_repeated_long_ngrams"]
        print(f"\n  2 docs sharing a 60-word span  →  repeated 50-grams: {n}  (detected ✓)")
        assert n >= 1


# ===========================================================================
# 14. Full pipeline integration
# ===========================================================================

class TestFullPipeline:

    REQUIRED_KEYS = {
        "totals", "length_stats", "statuses", "content_types",
        "languages", "lang_score_stats", "top_domains", "top_tlds",
        "top_suffixes", "duplicate_urls", "by_year", "by_month",
        "top_unigrams", "top_bigrams", "top_trigrams", "vocab",
        "exact_duplicates", "near_duplicates", "pii", "offensive",
        "self_contamination", "_doc_summaries",
    }

    def test_all_required_keys_present(self, analysis):
        missing = self.REQUIRED_KEYS - set(analysis.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_doc_summaries_count(self, analysis):
        assert len(analysis["_doc_summaries"]) == 1000

    def test_doc_summaries_fields(self, analysis):
        for doc in analysis["_doc_summaries"]:
            assert {"url", "lang", "lang_score", "n_tokens", "n_chars", "timestamp"} <= doc.keys()


# ===========================================================================
# 15. Performance & 100K scaling estimate
# ===========================================================================

class TestPerformanceEstimation:

    def test_stream_records_under_5s(self):
        t0 = time.perf_counter()
        recs = list(W.stream_records(CORPUS))
        elapsed = time.perf_counter() - t0
        print(f"\n[PERF] stream_records(1000 docs): {elapsed:.3f}s")
        assert elapsed < 5.0 and len(recs) == 1000

    def test_analyze_runtime_and_memory(self, records):
        import tracemalloc
        tracemalloc.start()
        t0 = time.perf_counter()
        result = W.analyze(records)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        scale = 100
        proj_s = elapsed * scale
        avg_bytes = sum(
            len((r["record"].get("text") or "").encode("utf-8"))
            for r in records
        ) / len(records)
        raw_text_gb = (avg_bytes * 100_000) / (1024 ** 3)
        proj_mem_gb = (peak_mb * scale) / 1024

        print(f"\n[PERF] analyze(1000 docs): {elapsed:.2f}s")
        print(f"[PERF] Peak memory: {peak_mb:.2f} MB")
        print("\n" + "=" * 60)
        print("SCALING ESTIMATE (100K RECORDS)")
        print("=" * 60)
        print(f"Runtime on 1K docs      : {elapsed:.2f} sec")
        print(f"Projected runtime 100K  : {proj_s:.1f} sec ({proj_s/60:.1f} min)")
        print(f"Peak memory on 1K docs  : {peak_mb:.1f} MB")
        print(f"Projected memory 100K   : {proj_mem_gb:.1f} GB (rough linear estimate)")
        print(f"Average document size   : {avg_bytes:.0f} bytes")
        print(f"Raw text size @100K     : {raw_text_gb:.1f} GB")
        print("\nNotes:")
        print("- Runtime estimate assumes near-linear scaling.")
        print("- MinHash and self-contamination may scale differently.")
        print("- Actual machine requirements should include safety margin.")
        print("=" * 60)

        assert result["totals"]["n_documents"] == 1000
        assert elapsed < 300.0
        assert proj_s > 0