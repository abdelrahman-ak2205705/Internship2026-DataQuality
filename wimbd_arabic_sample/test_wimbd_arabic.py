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
    pytest test_wimbd_arabic.py -v -W ignore::DeprecationWarning

Run one group:
    pytest test_wimbd_arabic.py -v -k "pii" -W ignore::DeprecationWarning

Show timing + scaling estimate:
    pytest test_wimbd_arabic.py -v --durations=10 -W ignore::DeprecationWarning
"""
from __future__ import annotations

import collections
import hashlib
import re
import sys
import time
import unittest.mock as mock
from pathlib import Path
import tracemalloc
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
            assert mark not in W.normalize_ar(f"ك{mark}تاب")

    def test_unifies_alef_hamza_above(self):
        assert W.normalize_ar("أحمد") == W.normalize_ar("احمد")

    def test_unifies_alef_hamza_below(self):
        assert W.normalize_ar("إسلام") == W.normalize_ar("اسلام")

    def test_unifies_alef_madda(self):
        assert W.normalize_ar("آمين") == W.normalize_ar("امين")

    def test_unifies_alef_maqsura(self):
        assert W.normalize_ar("مبنى") == W.normalize_ar("مبني")

    def test_unifies_ta_marbuta(self):
        assert W.normalize_ar("مدرسة") == W.normalize_ar("مدرسه")

    def test_removes_tatweel(self):
        assert "ـ" not in W.normalize_ar("جمـيل")

    def test_empty_string(self):
        assert W.normalize_ar("") == ""

    def test_latin_chars_unaffected(self):
        assert "Python" in W.normalize_ar("Python 3.12")


# ===========================================================================
# 2. Tokeniser
# ===========================================================================

class TestTokenize:

    def test_arabic_words(self):
        tokens = W.tokenize("الذكاء الاصطناعي")
        assert "الذكاء" in tokens and "الاصطناعي" in tokens

    def test_mixed_arabic_latin(self):
        tokens = W.tokenize("تعلم Python بسهولة")
        assert "Python" in tokens and "تعلم" in tokens

    def test_numbers_included(self):
        assert "2017" in W.tokenize("عام 2017")

    def test_punctuation_excluded(self):
        tokens = W.tokenize("مرحبا، كيف حالك؟")
        assert "،" not in tokens and "؟" not in tokens

    def test_empty_string(self):
        assert W.tokenize("") == []

    def test_real_doc_tokenizes_to_nonzero(self, records):
        text = records[0]["record"].get("text", "")
        assert len(W.tokenize(text)) > 0


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
        assert W.fingerprint(t) == W.fingerprint(t)

    def test_different_texts_differ(self):
        assert W.fingerprint("نص أول") != W.fingerprint("نص ثانٍ")

    def test_returns_32_char_hex(self):
        assert re.fullmatch(r"[0-9a-f]{32}", W.fingerprint("test"))

    def test_matches_hashlib_md5(self):
        text = "مرحبا"
        assert W.fingerprint(text) == hashlib.md5(text.encode("utf-8")).hexdigest()

    def test_whitespace_sensitive(self):
        assert W.fingerprint("a b") != W.fingerprint("a  b")


# ===========================================================================
# 4. PII detection
# ===========================================================================

class TestPIIPatterns:

    def _find(self, kind, text):
        return W.PII_PATTERNS[kind].findall(text)

    def test_email_basic(self):
        assert self._find("email", "user@example.com")

    def test_email_real_corpus_example(self):
        assert self._find("email", "ahmed_club2000@yahoo.com")

    def test_email_in_arabic_sentence(self):
        assert self._find("email", "راسلنا على info@kds.ae اليوم")

    def test_email_no_match_incomplete(self):
        assert not self._find("email", "user@") and not self._find("email", "@example.com")

    def test_ipv4_standard(self):
        assert self._find("ipv4", "192.168.1.1")

    def test_ipv4_real_corpus_example(self):
        assert self._find("ipv4", "217.218.48.21")

    def test_ipv4_no_match_three_octets(self):
        assert not self._find("ipv4", "192.168.1")

    def test_url_https(self):
        assert self._find("url", "https://example.com/path")

    def test_url_no_match_without_scheme(self):
        assert not self._find("url", "example.com/page")

    def test_phone_international(self):
        assert self._find("phone", "+966-555-1234")

    def test_phone_local(self):
        assert self._find("phone", "0501234567")

    def test_phone_arabic_indic_digits(self):
        assert self._find("phone", "٠٥٠١٢٣٤٥٦٧")


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
        n, _ = self._run(["هذا الرجل مثل كلب"])
        assert n == 1

    def test_clean_text_zero_hits(self):
        n, _ = self._run(["الذكاء الاصطناعي مفيد للغاية"])
        assert n == 0

    def test_normalization_applied_before_match(self):
        # diacritics on an offensive word must still match
        n, _ = self._run(["هذا كَلْبٌ"])
        assert n == 1

    def test_space_separated_multiple_words(self):
        n, terms = self._run(["كلب حمار"])
        assert n == 1 and len(terms) >= 2

    def test_waw_prefix_not_matched(self):
        n, _ = self._run(["وكلب"])
        assert n == 0

    def test_real_corpus_offensive_docs(self, analysis):
        assert analysis["offensive"]["n_docs_with_offensive"] == 30


# ===========================================================================
# 6. Exact duplicate detection
# ===========================================================================

class TestExactDuplicates:

    def test_three_dup_groups(self, analysis):
        assert analysis["exact_duplicates"]["n_dup_groups"] == 3

    def test_ten_docs_in_dup_groups(self, analysis):
        assert analysis["exact_duplicates"]["n_dup_docs"] == 10

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
        assert result["exact_duplicates"]["n_dup_groups"] == 0

    def test_fingerprint_differs_for_near_dups(self):
        assert W.fingerprint("نص أ") != W.fingerprint("نص ب")


# ===========================================================================
# 7. Near-duplicate detection (MinHash LSH)
# ===========================================================================

class TestNearDuplicates:

    def test_pair_count(self, analysis):
        assert analysis["near_duplicates"]["n_pairs"] == 109

    def test_cluster_count(self, analysis):
        assert analysis["near_duplicates"]["n_clusters"] == 47

    def test_threshold_and_perms(self, analysis):
        assert analysis["near_duplicates"]["threshold"] == 0.8
        assert analysis["near_duplicates"]["num_perm"] == 128

    def test_short_docs_not_indexed(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": "كلمة",
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(3)
        ]
        assert W.analyze(recs)["near_duplicates"]["n_pairs"] == 0

    def test_identical_long_docs_pair(self):
        long = "الذكاء الاصطناعي يعتمد على التعلم الآلي " * 10
        recs = [
            {"record": {"url": f"https://site{i}.com", "text": long,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i in range(2)
        ]
        assert W.analyze(recs)["near_duplicates"]["n_pairs"] >= 1

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
        assert W.analyze(recs)["near_duplicates"]["n_pairs"] == 0


# ===========================================================================
# 8. N-gram analysis
# ===========================================================================

class TestNgramAnalysis:

    def test_unigrams_populated(self, analysis):
        assert len(analysis["top_unigrams"]) > 0

    def test_unigrams_are_word_int_tuples(self, analysis):
        for word, count in analysis["top_unigrams"]:
            assert isinstance(word, str) and isinstance(count, int) and count > 0

    def test_no_stopwords_in_unigrams(self, analysis):
        for word, _ in analysis["top_unigrams"]:
            assert word not in W.ARABIC_STOPWORDS

    def test_bigrams_are_two_tokens(self, analysis):
        for bigram, _ in analysis["top_bigrams"]:
            assert len(bigram.split(" ")) == 2

    def test_trigrams_are_three_tokens(self, analysis):
        for trigram, _ in analysis["top_trigrams"]:
            assert len(trigram.split(" ")) == 3

    def test_vocab_size(self, analysis):
        assert analysis["vocab"]["n_types"] == 102115

    def test_normalization_merges_alef_variants(self):
        recs = [
            {"record": {"url": f"https://x.com/{i}", "text": t,
                        "normalized_url": None, "timestamp": None,
                        "content_type": None, "langdetect": None},
             "status": "PASSED"}
            for i, t in enumerate(["أحمد يدرس الهندسة", "احمد يدرس الهندسة"])
        ]
        unigrams = dict(W.analyze(recs)["top_unigrams"])
        assert unigrams.get("احمد", 0) == 2


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
        assert analysis["self_contamination"]["n_repeated_long_ngrams"] == 70778

    def test_ngram_size_is_50(self, analysis):
        assert analysis["self_contamination"]["ngram_size"] == 50

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
        assert W.analyze(recs)["self_contamination"]["n_repeated_long_ngrams"] == 0

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
        assert W.analyze(recs)["self_contamination"]["n_repeated_long_ngrams"] >= 1


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

import tracemalloc
import time
class TestPerformanceEstimation:

    def test_stream_records_under_5s(self):
        t0 = time.perf_counter()
        recs = list(W.stream_records(CORPUS))
        elapsed = time.perf_counter() - t0
        print(f"\n[PERF] stream_records(1000 docs): {elapsed:.3f}s")
        assert elapsed < 5.0
        assert len(recs) == 1000

    def test_analyze_runtime_and_memory(self, records):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = W.analyze(records)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        print(f"\n[PERF] analyze(1000 docs): {elapsed:.2f}s")
        print(f"[PERF] Peak memory: {peak_mb:.2f} MB")
        assert result["totals"]["n_documents"] == 1000

    def test_project_to_100k(self, records):
        tracemalloc.start()
        t0 = time.perf_counter()
        W.analyze(records)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        scale = 100  # 1K -> 100K
        projected_runtime_sec = elapsed * scale
        projected_runtime_min = projected_runtime_sec / 60
        peak_mb = peak / (1024 * 1024)
        projected_memory_gb = (peak_mb * scale) / 1024
        avg_doc_bytes = (
            sum(
                len((r["record"].get("text") or "").encode("utf-8"))
                for r in records
            )
            / len(records)
        )
        raw_text_gb = (
            avg_doc_bytes * 100_000
        ) / (1024 ** 3)
        print("\n" + "=" * 60)
        print("SCALING ESTIMATE (100K RECORDS)")
        print("=" * 60)
        print(f"Runtime on 1K docs       : {elapsed:.2f} sec")
        print(
            f"Projected runtime 100K  : "
            f"{projected_runtime_sec:.1f} sec "
            f"({projected_runtime_min:.1f} min)"
        )
        print(f"Peak memory on 1K docs  : {peak_mb:.1f} MB")
        print(
            f"Projected memory 100K   : "
            f"{projected_memory_gb:.1f} GB "
            f"(rough linear estimate)"
        )
        print(f"Average document size   : {avg_doc_bytes:.0f} bytes")
        print(f"Raw text size @100K     : {raw_text_gb:.1f} GB")
        print("\nNotes:")
        print("- Runtime estimate assumes near-linear scaling.")
        print("- MinHash and self-contamination may scale differently.")
        print("- Actual machine requirements should include safety margin.")
        print("=" * 60)
        assert projected_runtime_sec > 0
