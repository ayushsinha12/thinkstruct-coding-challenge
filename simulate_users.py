import argparse, random, time, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from search_cli import BM25Index, load_json_files, proximity_rerank, passes_filters, split_csv_words

# A list of example search queries
QUERIES = [
    "wheel assembly", "spindle repair apparatus", "multi-piece wheel frame",
    "braking assembly segmented disc", "swivel castor", "motorcycle wheel adapter",
    "bassinet wheel", "hub flange", "spoke injection molding", "composite wheel plate"
]

def one_search(i: int, records, index, two_phase: bool, phase1_k: int) -> float:
    """
    Runs a simulated search on the patent index, optionally performing a two-phase rerank, 
    and returns the time taken.

    Args:
        i (int): Identifier for the search instance (not used in logic).
        records (list): Loaded patent record data.
        index (BM25Index): BM25 search index.
        two_phase (bool): Whether to perform two-phase search reranking.
        phase1_k (int): Number of top results to keep before reranking.
    Returns:
        float: Time taken for the search in seconds.
    """
    q = random.choice(QUERIES)
    t0 = time.perf_counter()

    # Simple baseline: no filters to stress the scorer
    candidates = list(range(len(records)))
    ranked1 = index.score_subset(q, candidates)
    if two_phase:
        ranked1 = ranked1[:max(1, phase1_k)]
        _ = proximity_rerank(records, ranked1, q)
    return time.perf_counter() - t0

def main():
    #Initalizing simulation
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--phase1-k", type=int, default=100)
    args = ap.parse_args()

    # Loading data
    print("Loading data/index…")
    records = load_json_files(pathlib.Path(args.data_dir))

    # Building BM25 index
    index = BM25Index(records, (3.0,2.0,2.0,1.0))
    print(f"Docs={len(records)}. Simulating {args.users} users… two={args.two_phase}")

    # Running the simulation
    latencies: List[float] = []
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as ex:
        futs = [ex.submit(one_search, i, records, index, args.two_phase, args.phase1_k)
                for i in range(args.users)]
        for f in as_completed(futs):
            latencies.append(f.result())
    total = time.perf_counter() - t_all

    latencies.sort()
    p50 = latencies[int(0.50*len(latencies))]
    p95 = latencies[int(0.95*len(latencies))-1]
    print(f"Throughput: {args.users/total:.1f} req/s | total={total:.2f}s")
    print(f"Latency p50={p50:.3f}s  p95={p95:.3f}s  max={latencies[-1]:.3f}s")

if __name__ == "__main__":
    main()