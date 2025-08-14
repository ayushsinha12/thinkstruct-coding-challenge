"""
Thinkstruct Patent Search – Part 1 (CLI)
- Natural language query + optional filters (hybrid)
- BM25 ranking over: title, abstract, claims, detailed_description
- Minimal deps (stdlib only), works offline on provided JSON

Examples:

  # Hybrid with class prefix and abstract keywords
  python search_cli.py --data-dir ./patent_data_small --query "multi-piece wheel frame" --class-prefix B60 --top-k 5 --time

  # Hybrid with title keyword AND logic + show best passage
  python search_cli.py --data-dir ./patent_data_small --query "spindle repair apparatus" --title-kw "spindle,repair" --show-passages

  # Hybrid with EXACT title match
  python search_cli.py --data-dir ./patent_data_small --query "spindle repair apparatus" --exact-title "SPINDLE REPAIR APPARATUS AND METHOD"

  # Two-phase rerank on top 100 candidates
  python search_cli.py --data-dir ./patent_data_small --query "braking assembly segmented disc" --two-phase --phase1-k 100 --top-k 5 --time

  # JSON (NDJSON) output
  python search_cli.py --data-dir ./patent_data_small --query "braking assembly segmented disc" --json --top-k 3
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Optional

# ---------------- Tokenization ----------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE) # creates a regular expression to match any sequence of alphanumeric characters

def tokenize(text: str) -> List[str]:
    """
    Returns a list of string tokens.

    Args:
        str: Input text to tokenize.
    Returns:
        List[str]: Lowercased alphanumeric tokens found in the text.
    """
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]

# Stores values from json in lookup dictionary
FIELD_ALIASES = {
    "title": {"title"},
    "doc_number": {"doc_number", "document number", "number"},
    "abstract": {"abstract"},
    "detailed_description": {"detailed_description", "description", "detailed description"},
    "claims": {"claims"},
    "bibtex": {"bibtex", "bibtext", "bibtex citation"},
    "classification": {"classification", "classification code"},
    "filename": {"filename"},
}

def standardize_key(key: str) -> str:
    """
    Standardize key name.

    Args:
        key: Raw field name from JSON.
    Returns:
        str: Normalized field name (canonical or lowercased/trimmed).
    """
    normalized_key = key.strip().lower()
    for stand_vers, alt_names in FIELD_ALIASES.items():
        if normalized_key in alt_names:
            return stand_vers
    return normalized_key

def join_strings(data: Any) -> str:
    """
    Converts a string or list of strings into a single newline-separated string.

    Args:
        data: A string or a list of strings (others are ignored).
    Returns:
        str: Joined newline-separated string, or empty string if none.
    """
    if isinstance(data, list):
        return "\n".join(s for s in data if isinstance(s, str))
    if isinstance(data, str):
        return data
    return ""

def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns fully cleaned, and standardized record.

    Args:
        raw: A raw patent record dictionary from JSON.

    Returns:
        Dict[str, Any]: Record with canonical keys and normalized text fields.
    """
    rec = {}
    for k, v in raw.items():
        rec[standardize_key(k)] = v

    # Making sure these keys exist; set to "" if missing
    rec.setdefault("title", "")
    rec.setdefault("doc_number", "")
    rec.setdefault("abstract", "")

    rec["detailed_description"] = join_strings(rec.get("detailed_description", ""))
    rec["claims"] = join_strings(rec.get("claims", ""))
    rec.setdefault("classification", "")
    rec.setdefault("filename", "")
    return rec


def load_json_files(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads all 'patents_*.json' files from the given directory, normalizes each record,
    and returns a List of cleaned patent dictionaries. Exits if no valid data is found.

    Args:
        data_dir: Directory containing patents_*.json files.
    Returns:
        List[Dict[str, Any]]: Cleaned patent records.
    """
    records: List[Dict[str, Any]] = []

    # Get all files matching the pattern and sort them
    files = sorted(list(data_dir.glob("patents_*.json")))
    if not files:
        print(f"[error] No files matching 'patents_*.json' in {data_dir}", file=sys.stderr)
        sys.exit(1)
    for fp in files:
        try:
            # Read and parse the JSON file
            with open(fp, "r", encoding="utf-8") as f:
                arr = json.load(f)
            # Making sure it is a list of dictionaries
            if isinstance(arr, list):
                for item in arr:
                    # Normalize key and value of valid dictionarys only
                    if isinstance(item, dict):
                        records.append(normalize_record(item))
        except Exception as e:
            print(f"[warn] Failed to read {fp.name}: {e}", file=sys.stderr)
    if not records:
        print(f"[error] No records loaded from {data_dir}", file=sys.stderr)
        sys.exit(1)
    return records

# ---------------- Filters ----------------

def split_csv_words(csv_string: str) -> List[str]:
    """
    Split comma-separated text into a List of lowercase words.

    Args:
        csv_string: Comma-separated words (e.g., "foo, bar").
    Returns:
        List[str]: Trimmed, lowercased words (empty if input is empty).
    """
    if not csv_string:
        return []
    return [w.strip().lower() for w in csv_string.split(",") if w.strip()]


def passes_filters(
    rec: Dict[str, Any],
    class_prefix: str,
    title_kw: List[str],
    abstract_kw: List[str],
    title_exact: str,
) -> bool:
    """
    Checks if a singular patent passes all the selected filters.

    Args:
        rec: Patent record to test.
        class_prefix: Required classification prefix (case-insensitive).
        title_kw: Required substrings that must all appear in title.
        abstract_kw: Required substrings that must all appear in abstract.
        title_exact: Exact title string to match (case-insensitive).

    Returns:
        bool: True if the record passes all filters; otherwise False.
    """

    # Classification prefix (e.g., B60B)
    if class_prefix:
        if not rec.get("classification", "").upper().startswith(class_prefix.upper()):
            return False

    # Exact title match (case-insensitive, trimmed)
    if title_exact:
        if rec.get("title", "").strip().lower() != title_exact.strip().lower():
            return False

    # Title keywords (AND)
    title_l = rec.get("title", "").lower()
    for w in title_kw:
        if w not in title_l:
            return False

    # Abstract keywords (AND)
    abstract_l = rec.get("abstract", "").lower()
    for w in abstract_kw:
        if w not in abstract_l:
            return False

    return True

# ---------------- BM25 Index (Phase-1) ----------------
class BM25Index:
    def __init__(self, records: List[Dict[str, Any]],
                 weights: Tuple[float, float, float, float], # adjust weights of each field's contribution
                 k1: float = 1.5, b: float = 0.75): # BM25 hyperparameters (term saturation, length normalization)
        """
        Build a multi-field BM25 index over title/abstract/claims/description.

        Args:
            records: Cleaned patent records to index.
            weights: Field weights (title, abstract, claims, description).
            k1: BM25 term-frequency saturation parameter.
            b: BM25 length-normalization parameter.
        Returns:
            None
        """
        self.records = records
        self.k1 = k1
        self.b = b
        self.fields = ("title", "abstract", "claims", "detailed_description")
        self.weights = dict(zip(self.fields, weights)) # mapping each field to its weight
        
        # Storing core stats
        self.doc_freq: Dict[str, Dict[str, int]] = {f: defaultdict(int) for f in self.fields} # in how many documents that field contains the term 
        self.doc_lengths: Dict[str, List[int]] = {f: [] for f in self.fields} # length of document in that field
        self.term_freqs: Dict[str, List[Counter]] = {f: [] for f in self.fields} # frequency of term in document

        # Building those stats
        for rec in self.records:
            for f in self.fields:
                toks = tokenize(rec.get(f, ""))
                term_counts = Counter(toks)
                self.term_freqs[f].append(term_counts)
                length = sum(term_counts.values())
                self.doc_lengths[f].append(length)
                for term in term_counts:
                    self.doc_freq[f][term] += 1
        self.total_docs = len(self.records)
        self.avg_doc_len = {f: (sum(self.doc_lengths[f]) / max(1, self.total_docs)) for f in self.fields}


    def _idf(self, f: str, term: str) -> float:
        """
        Computes the BM25 inverse document frequency for a term, giving higher scores to rarer terms.

        Args:
            f: Field name.
            term: Token to score.
        Returns:
            float: Smoothed BM25 IDF value.
        """
        n = self.doc_freq[f].get(term, 0)
        return math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1e-9)

    
    def score_subset(self, query: str, candidates: Iterable[int]) -> List[Tuple[int, float, str]]:
        """
        Ranks a set of documents against a query using BM25, combining scores from multiple fields with
        weights, and returns them ordered by relevance.

        Args:
            query: Natural-language query.
            candidates: Iterable of candidate doc indices to score.
        Returns:
            List[Tuple[int, float, str]]: (doc_id, score, best_field) sorted by score desc.
        """
        q_terms = tokenize(query)
        if not q_terms:
            return []
        
        # Accumulates total BM25 score per doc_id across all fields
        doc_scores: Dict[int, float] = {}

        # Tracks which field contributed most to score
        best_field_for_doc: Dict[int, Tuple[str, float]] = {}

        # Score each field separately, then combine with field weights
        for fname in self.fields:
            fweight = self.weights[fname]
            if fweight <= 0:
                continue

            avgdl = self.avg_doc_len[fname] or 1.0
            for i in candidates:
                term_freq_map = self.term_freqs[fname][i]
                doc_len = self.doc_lengths[fname][i] or 1
                field_sum = 0.0

                for t in q_terms:
                    term_freq = term_freq_map.get(t, 0)
                    if term_freq == 0:
                        continue

                    # BM25 formula: combines term frequency, document length normalization, and IDF
                    idf = self._idf(fname, t)
                    denom = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                    field_sum += idf * (term_freq * (self.k1 + 1)) / denom
                if field_sum:
                    total = doc_scores.get(i, 0.0) + fweight * field_sum
                    doc_scores[i] = total

                    # Track the field that gave the highest contribution for this document
                    if (i not in best_field_for_doc) or (field_sum > best_field_for_doc[i][1]):
                        best_field_for_doc[i] = (fname, field_sum)
        
        # Sort documents by total score (highest first) and return with best contributing field                
        ranked = sorted(((doc_id, score, best_field_for_doc.get(doc_id, ("", 0.0))[0]) for doc_id, score in doc_scores.items()),
                        key=lambda x: x[1], reverse=True)
        return ranked

# ---------------- Phase‑2: Proximity + Phrase reranker ----------------

def _bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Takes a list of words (tokens) and pairs each word with the one right after it.

    Args:
        tokens: Token list.
    Returns:
        List[Tuple[str, str]]: Adjacent token pairs (2-grams).
    """
    return list(zip(tokens, tokens[1:]))


def _coverage_window_score(tokens: List[str], q_terms: List[str], window: int = 12) -> Tuple[float, Tuple[int, int]]:
    """
    Calculate the highest query term coverage within any fixed-size sliding window of tokens.

    Args:
        tokens: Text tokens.
        q_terms: Query tokens.
        window: Window size in tokens.
    Returns:
        Tuple[float, Tuple[int, int]]: (best_coverage_ratio, (start_idx, end_idx)).
    """
    if not tokens or not q_terms:
        return 0.0, (0, 0)
    query_set = set(q_terms)
    best_cov = 0.0
    best_window = (0, 0)
    counts = Counter()
    left_idx = 0
    unique = 0

    # Expand the window by moving the right pointer
    for right_idx, tok in enumerate(tokens):
        if tok in query_set:
            counts[tok] += 1
            if counts[tok] == 1:
                unique += 1
        # Shrink the window if it exceeds the allowed size
        while (right_idx - left_idx + 1) > window:
            left_tok = tokens[left_idx]
            if left_tok in query_set:
                counts[left_tok] -= 1
                if counts[left_tok] == 0:
                    unique -= 1
            left_idx += 1
        # Calculate coverage ratio for the current window
        cov_rat = unique / max(1, len(query_set))
        if cov_rat > best_cov:
            best_cov = cov_rat
            best_window = (left_idx, right_idx)
    return best_cov, best_window


def _field_rerank_score(text: str, q_terms: List[str]) -> Tuple[float, Optional[str]]:
    """
    Gives a boost score for how well a single text field matches the user query.

    Args:
        text: Field text to evaluate.
        q_terms: Tokenized query terms.
    Returns:
        Tuple[float, Optional[str]]: (boost_score, optional best local snippet).
    """
    if not text:
        return 0.0, None
    text_low = text.lower()
    toks = tokenize(text)
    score = 0.0

    # Boosted score for exact whole-query phrase match
    phrase = " ".join(q_terms)
    if phrase and phrase in text_low:
        score += 5.0

    # Boosted score for bigrams (adjacent word pairs) from query that appear in the text
    query_bi = _bigrams(q_terms)
    if query_bi:
        text_bi = set(_bigrams(toks))
        score += 1.25 * sum(1 for bigram in query_bi if bigram in text_bi)

    # Bossted score for how tightly query terms cluster (coverage inside sliding window)
    cov, span = _coverage_window_score(toks, q_terms, window=12)
    score += 6.0 * cov
    snippet = " ".join(toks[max(span[0]-5, 0): span[1]+6]) if span != (0, 0) else None
    return score, snippet


def proximity_rerank(records: List[Dict[str, Any]],
                     candidates_ranked: List[Tuple[int, float, str]],
                     query: str) -> List[Tuple[int, float, str]]:
    """
    Re-ranks candidate documents by combining their initial BM25 scores with
    proximity-based scores from specific fields, giving extra weight to matches in titles.

    Args:
        records: Patent records (for accessing fields).
        candidates_ranked: Phase-1 results as (doc_id, bm25_score, best_field).
        query: Original query string.

    Returns:
        List[Tuple[int, float, str]]: Re-ranked (doc_id, total_score, best_field).
    """
    if not candidates_ranked:
        return []
    q_terms = tokenize(query) # Tokenize the query into terms for proximity scoring
    out: List[Tuple[int, float, str]] = []
    for doc_id, bm25_score, best_field1 in candidates_ranked:
        rec = records[doc_id]
        best_field2 = best_field1
        best_local = -1.0 
        best_snip = None
        total = bm25_score 
        for f in ("title", "abstract", "claims", "detailed_description"):
             # Compute proximity score for this field
            s, sn = _field_rerank_score(rec.get(f, ""), q_terms)
            w = 2.5 if f == "title" else 1.0 
            total += w * s
            
            # Update best local match if this field's score is higher
            if s > best_local:
                best_local = s
                best_field2 = f
                best_snip = sn

        rec["_phase2_snippet"] = best_snip
        out.append((doc_id, total, best_field2))

    # Sort by total score in descending order
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------- Output ----------------

def highlight(text: str, q_terms: List[str]) -> str:
    """
    Wraps all occurrences of query terms with **.

    Args:
        text: Source text.
        q_terms: Query tokens to highlight.

    Returns:
        str: Text with **term**.
    """
    out = text
    for t in sorted(set(q_terms), key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(t)})\b", r"**\1**", out)
    return out


def best_snippet_from(text: str, q_terms: List[str], max_len: int = 260) -> str:
    """
    Finds the part of a text containing the earliest query term 
    and returns a shortened snippet around it.

    Args:
        text: Source text.
        q_terms: Query tokens.
        max_len: Max snippet length in characters.
    Returns:
        str: Trimmed, highlighted snippet (or empty string).
    """
    if not text:
        return ""
    lower = text.lower()
    positions = []

    # going through every unique query term 
    for t in set(q_terms):
        p = lower.find(t) # first occurence of each term
        if p != -1:
            positions.append(p)
    if not positions:
        return text[:max_len]
    
    # Snippet starts about 1/3 of the way before the first occureence of a query term
    start = max(0, min(positions) - max_len // 3)
    end = min(len(text), start + max_len)
    segment = text[start:end] # Extracts the relevant snippet of the text
    return ("..." if start > 0 else "") + highlight(segment, q_terms) + ("..." if end < len(text) else "")

def make_default_snippet(rec: Dict[str, Any], q_terms: List[str]) -> str:
    """
    Returns the best matching snippet from a patent’s text for the given query.

    Args:
        rec: Patent record containing fields.
        q_terms: Query tokens.
    Returns:
        str: First non-empty snippet from abstract/claims/description, or empty string.
    """

    # iterates through those fiels in priority order and stops until finding
    for field in ("abstract", "claims", "detailed_description"):
        seg = best_snippet_from(rec.get(field, ""), q_terms)
        if seg:
            return seg
    return ""

def print_hit(rank: int, score: float, rec: Dict[str, Any], q_terms: List[str],
              as_json: bool, best_field: str, show_passages: bool) -> None:
    """
    Pretty-prints one search result (or emits JSON), including a snippet and optional best passage.

    Args:
        rank: 1-based rank in the results list.
        score: Document score.
        rec: Patent record to print.
        q_terms: Query tokens for highlighting.
        as_json: If True, print as NDJSON instead of pretty text.
        best_field: Field name of strongest match.
        show_passages: If True, include best passage from best_field.
    Returns:
        None
    """

    base = {
        "rank": rank,
        "score": round(float(score), 4),
        "title": rec.get("title", ""),
        "doc_number": rec.get("doc_number", ""),
        "classification": rec.get("classification", ""),
        "filename": rec.get("filename", ""),
        "snippet": make_default_snippet(rec, q_terms),
    }

    if show_passages and best_field:
        field_text = rec.get(best_field, "")
        base["best_field"] = best_field
        base["best_passage"] = best_snippet_from(field_text, q_terms)

    if as_json:
        print(json.dumps(base, ensure_ascii=False))
        return
    
    print(f"[{rank}] score={base['score']:.4f} | {base['title'] or '(no title)'}")
    meta = []
    if base["doc_number"]: meta.append(f"doc {base['doc_number']}")
    if base["classification"]: meta.append(f"class {base['classification']}")
    if base["filename"]: meta.append(base["filename"])
    if meta:
        print("    " + " | ".join(meta))
    print("    " + base["snippet"])
    if show_passages and best_field and base.get("best_passage"):
        print(f"    [top passage in {best_field}] {base['best_passage']}")
    print()

# ---------------- Main ----------------

def parse_weights(s: str) -> Tuple[float, float, float, float]:
    """
    Parses the value of the weightage for different parts of the patent.

    Args:
        s: Comma-separated four numbers (e.g., "3,2,2,1").
    Returns:
        Tuple[float, float, float, float]: Weights for (title, abstract, claims, description).
    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    try:
        a = tuple(float(x) for x in s.split(","))
        if len(a) != 4:
            raise ValueError
        return a  # title, abstract, claims, description
    except Exception:
        raise argparse.ArgumentTypeError("weights must be four comma-separated numbers, e.g., 3,2,2,1")

def main():

    # All the various command line arguments for CLI
    ap = argparse.ArgumentParser(description="Thinkstruct Patent Search (BM25 CLI)")
    ap.add_argument("--data-dir", type=str, required=True, help="Folder with patents_*.json")
    ap.add_argument("--query", type=str, required=True, help="Natural-language query")
    ap.add_argument("--top-k", type=int, default=10, help="Results to show")
    ap.add_argument("--weights", type=parse_weights, default=(3.0, 2.0, 2.0, 1.0),
                    help="Field weights: title,abstract,claims,description (e.g., 3,2,2,1)")
    ap.add_argument("--json", action="store_true", help="Output JSON lines")
    ap.add_argument("--show-passages", action="store_true", help="Show top passage & field source")

    # Hybrid filters
    ap.add_argument("--class-prefix", type=str, default="",
                    help="Filter: classification starts with (e.g., B60B)")
    ap.add_argument("--title-kw", type=str, default="",
                    help='Filter: CSV words that must be in TITLE (e.g., "spoke,vent")')
    ap.add_argument("--abstract-kw", type=str, default="",
                    help='Filter: CSV words that must be in ABSTRACT (e.g., "cooling,vent")')

    # Preferred exact-title flag 
    ap.add_argument("--exact-title", "--exactTitle", dest="exact_title", type=str, default="",
                    help='Filter: TITLE must exactly equal this string (case-insensitive)')
    
    # Baseline / timing
    ap.add_argument("--time", action="store_true",
                    help="Print timing breakdown (load/index/filter/score) and doc counts.")

    # Two-phase searching
    ap.add_argument("--two-phase", action="store_true",
                    help="Enable phase‑2 proximity/phrase rerank on top‑K BM25 candidates.")
    ap.add_argument("--phase1-k", type=int, default=100,
                    help="How many top BM25 candidates to rerank in phase‑2.")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[error] data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Timings ---
    t0 = time.perf_counter()
    records = load_json_files(data_dir)
    t_load = time.perf_counter()

    # Build index (separate timing)
    index = BM25Index(records, args.weights)
    t_index = time.perf_counter()

    # Determine candidates
    title_kw = split_csv_words(args.title_kw)
    abstract_kw = split_csv_words(args.abstract_kw)

    # Unify new and legacy exact-title flags
    exact_title = args.exact_title

    candidates = [i for i, rec in enumerate(records) if passes_filters(
        rec, args.class_prefix, title_kw, abstract_kw, exact_title
    )]

    t_filter = time.perf_counter()

    if not candidates:
        print("[info] No documents matched the filters. Tip: relax --class-prefix / --title-kw / --abstract-kw / --exact-title.")
        if args.time:
            print(f"[time] load={t_load - t0:.3f}s | index={t_index - t_load:.3f}s | "
                  f"filter={t_filter - t_index:.3f}s | score=0.000s | total={t_filter - t0:.3f}s")
            print(f"[time] N_docs={len(records)} | N_candidates=0")
        return

    # Phase‑1 BM25
    ranked_phase1 = index.score_subset(args.query, candidates)
    t_score = time.perf_counter()

    # If BM25 had no lexical overlap but filters selected things (e.g., exact title),
    # still pass some docs forward with zero score so user sees them.
    if not ranked_phase1 and candidates:
        ranked_phase1 = [(i, 0.0, "") for i in candidates]

    # Phase‑2 rerank (optional)
    if args.two_phase and ranked_phase1:
        cand = ranked_phase1[: max(1, args.phase1_k)]
        ranked_final = proximity_rerank(records, cand, args.query)
    else:
        ranked_final = ranked_phase1
    t_score2 = time.perf_counter()

    if args.time:
        score1 = t_score - t_filter              # Phase‑1 BM25 time
        score2 = t_score2 - t_score              # Phase‑2 rerank time (0 if not used)
        total  = (t_score2 if args.two_phase else t_score) - t0

        if args.two_phase:
            print(f"[time] load={t_load - t0:.3f}s | index={t_index - t_load:.3f}s | "
                  f"filter={t_filter - t_index:.3f}s | score1={score1:.3f}s | "
                  f"score2={score2:.3f}s | total={total:.3f}s")
        else:
            print(f"[time] load={t_load - t0:.3f}s | index={t_index - t_load:.3f}s | "
                  f"filter={t_filter - t_index:.3f}s | score={score1:.3f}s | total={total:.3f}s")

        print(f"[time] N_docs={len(records)} | N_candidates={len(candidates)}")

    if not ranked_final:
        print("[info] Empty query or no matches.")
        return

    q_terms = tokenize(args.query)
    shown = 0
    for doc_id, score, best_field in ranked_final:
        print_hit(shown + 1, score, records[doc_id], q_terms, args.json, best_field, args.show_passages)
        shown += 1
        if shown >= args.top_k:
            break

if __name__ == "__main__":
    main()