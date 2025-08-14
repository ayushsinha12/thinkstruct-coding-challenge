import json, time, pathlib, re, shlex
from typing import List, Dict, Tuple, Any, Optional
from search_cli import (
    BM25Index, load_json_files, passes_filters, split_csv_words,
    proximity_rerank, tokenize, print_hit
)

HIST_DIR = pathlib.Path.home() / ".thinkstruct"
HIST_DIR.mkdir(exist_ok=True)

def save_history(user: str, entry: Dict[str, Any]) -> None:
    """
    Appends a JSON-formatted history entry for a given user to their history file.

    Args:
        user: Username whose history file will be appended to.
        entry: Search details to save.
    Returns:
        None
    """
    p = HIST_DIR / f"{user}_history.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_history(user: str, last: int = 10) -> List[Dict[str, Any]]:
    """
    Reads and returns the saved search history from a user.

    Args:
        user: Username whose history file will be read.
        last: Number of most recent entries to return.
    Returns:
        List[Dict[str, Any]]: Parsed history entries.
    """
    p = HIST_DIR / f"{user}_history.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-last:]
    return [json.loads(x) for x in lines]


def run_search(records, index: BM25Index, *,
               query: str, top_k: int = 5,
               class_prefix: str = "", title_kw: str = "",
               abstract_kw: str = "", title_exact: str = "",
               two_phase: bool = False, phase1_k: int = 100) -> List[Tuple[int, float, str]]:
    """
    Executes a two-phase BM25 search with optional filtering and proximity reranking.

    Args:
        records: Patent records to search.
        index: Pre-built BM25Index.
        query: Natural-language query string.
        top_k: Number of results to return.
        class_prefix: Classification code prefix filter.
        title_kw: CSV keywords required in the title.
        abstract_kw: CSV keywords required in the abstract.
        title_exact: Exact title match filter.
        two_phase: Whether to apply phase-2 reranking.
        phase1_k: Number of candidates for phase-2 reranking.
    Returns:
        List[Tuple[int, float, str]]: Ranked (doc_id, score, best_field).
    """

    # Apply filters → BM25 (phase-1) → optional proximity/phrase rerank (phase-2).
    title_kw_list = split_csv_words(title_kw)
    abstract_kw_list = split_csv_words(abstract_kw)

    # Only keeps records that pass the filters
    candidates = [i for i, rec in enumerate(records) if passes_filters(
        rec, class_prefix, title_kw_list, abstract_kw_list, title_exact
    )]
    if not candidates:
        return []

    # Scores the filtered candidates with BM25 and returns a list of tuples,
    # sorted by score
    ranked_phase1 = index.score_subset(query, candidates)

    if not ranked_phase1 and candidates:
        ranked_phase1 = [(i, 0.0, "") for i in candidates]

    # Runs two phase search
    if two_phase and ranked_phase1:
        rerank_input = ranked_phase1[:max(1, phase1_k)]
        ranked_final = proximity_rerank(records, rerank_input, query)
    else:
        ranked_final = ranked_phase1

    return ranked_final[:top_k]


def pretty_print(records, ranked, query: str):
    """
    Displays ranked search results with query term highlighting and best matching field.

    Args:
        records: All patent records.
        ranked: Ranked results as (doc_id, score, best_field).
        query: Original query string.
    Returns:
        None
    """

    q_terms = tokenize(query)
    for i, (doc_id, score, best_field) in enumerate(ranked, start=1):
        print_hit(i, score, records[doc_id], q_terms, False, best_field, True)


def get_quoted_query(raw: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parses a string for a double-quoted query and optional parameters.

    Args:
        raw: Input string starting with a quoted query.
    Returns:
        Optional[Tuple[str, List[str]]]: (query_without_quotes, option_tokens) or None if not valid.
    """
    m = re.match(r'^\s*"([^"]+)"\s*(.*)$', raw)
    if not m:
        print('Error: Query must be enclosed in double quotes, e.g., "wheel assembly"')
        return None

    query = m.group(1)              # inside the "..."
    rest  = m.group(2).strip()      # tokens after the closing quote
    opt_parts = shlex.split(rest) if rest else []
    return query, opt_parts

def main():
    #Initalizing search engine
    data_dir = input("Data directory (e.g. ./patent_data_small): ").strip() or "./patent_data_small"
    user = input("Username for session history: ").strip() or "default"

    # Loading patent data
    print("Loading data & building index…")
    t0 = time.perf_counter()
    records = load_json_files(pathlib.Path(data_dir))

    # Building BM25 index
    index = BM25Index(records, (3.0, 2.0, 2.0, 1.0))
    print(f"Ready. {len(records)} docs | init={time.perf_counter()-t0:.2f}s\n")

    # Displays available commands
    print("Commands:")
    print("  help         → help")
    print("  history           → show your last 10 searches")
    print("  quit             → exit")
    print("Filters syntax    → class=B60  titlekw=spoke,vent  abskw=cooling  title=\"EXACT TITLE\"")
    print("Two-phase flags   → two=1  or  two-phase=on  or  two-phase on   (p1k=100 sets the rerank pool)")
    print('Example: "wheel assembly" class=B60 two=1 p1k=80 top=5\n')

    while True:
        raw = input("thinkstruct> ").strip()
        if not raw:
            continue
        if raw == "quit":
            break
        if raw in ("help"):
            print('Enter: "<query>" [class=..] [titlekw=..] [abskw=..] [title="exact"] [two=1] [two-phase=on] [p1k=100] [top=5]')
            continue
        if raw in ("history"):
            hist = read_history(user)
            if not hist:
                print("(no history yet)")
            else:
                for h in hist:
                    print(f"- {h['ts']} | q='{h.get('query','')}' | top={h.get('top_k',5)} "
                          f"| two={h.get('two_phase', False)} | p1k={h.get('phase1_k',100)} "
                          f"| class={h.get('class_prefix','')}")
            continue

        # Remove accidental leading 'search ' or 'filter '
        low = raw.lower()
        if low.startswith("search "):
            raw = raw.split(" ", 1)[1]
        elif low.startswith("filter "):
            raw = raw.split(" ", 1)[1]

        # Validate and extract quoted query
        result = get_quoted_query(raw)
        if not result:
            continue
        q, opt_parts = result

        # Parse flags
        opts = dict(class_prefix="", title_kw="", abstract_kw="", title_exact="",
                    top_k=5, two_phase=False, phase1_k=100)
        i = 0
        while i < len(opt_parts):
            p = opt_parts[i]
            if "=" in p:
                k, v = p.split("=", 1)
                if k == "class":
                    opts["class_prefix"] = v
                elif k == "titlekw":
                    opts["title_kw"] = v
                elif k == "abskw":
                    opts["abstract_kw"] = v
                elif k == "title":
                    opts["title_exact"] = v.strip('"').strip("'")
                elif k == "top":
                    opts["top_k"] = int(v)
                elif k in ("two", "two-phase", "two_phase"):
                    opts["two_phase"] = v.lower() in ("1", "true", "yes", "on")
                elif k in ("p1k", "phase1_k"):
                    opts["phase1_k"] = int(v)
            else:
                if p in ("two-phase", "two_phase") and (i + 1) < len(opt_parts):
                    nxt = opt_parts[i + 1].lower()
                    opts["two_phase"] = nxt in ("1", "true", "yes", "on")
                    i += 1
            i += 1

        # Run search
        t1 = time.perf_counter()
        ranked = run_search(records, index, query=q, **opts)
        dt = time.perf_counter() - t1

        if not ranked:
            print(f"[time] {dt:.3f}s  | two-phase={opts['two_phase']} | k={opts['top_k']} | p1k={opts['phase1_k']}")
            print("[info] No documents matched the filters. Tip: adjust class= / titlekw= / abskw= / title=\"...\".")
            continue

        print(f"[time] {dt:.3f}s  | two-phase={opts['two_phase']} | k={opts['top_k']} | p1k={opts['phase1_k']}")
        pretty_print(records, ranked, q)

        save_history(user, {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": q, **opts
        })

if __name__ == "__main__":
    main()