# Thinkstruct Patent Search (Backend Project)

## Problem Statement (what I chose to solve)
A fast, offline search tool over 2024–present vehicle patent application JSON files that:
1) accepts a natural-language query,
2) ranks patents by relevance using multiple fields (Title, Abstract, Claims, Detailed Description),
3) supports **hybrid filtering** (classification code prefix, title/abstract keyword constraints, exact-title match),
4) optionally applies a **second-phase proximity/phrase re-ranking** to improve the top results, and
5) provides a simple **interactive interface** with user session history and the ability to **simulate concurrent users**.

---

## How the code addresses the problem (and which enhancement(s) I picked & why)

### Core approach
- **Data loading & normalization** (`search_cli.py`)
  - Loads `patents_*.json` from a directory.
  - Normalizes keys (e.g., “Detailed Description” → `detailed_description`).
  - Joins list fields into text; safely defaults any missing field to `""`.
  - **Missing fields** were handled by keeping the patents with the missing fields and treating the missing fields as empty strings so they remain searchable without crashing or skewing stats.

- **Lexical ranking over multiple fields (Phase-1)** (`search_cli.py → BM25Index`)
  - BM25 is a ranking algorithm that scores documents by how closely their terms match a search query, adjusting for term frequency and document length.
  - Implements BM25 over four fields: `title`, `abstract`, `claims`, `detailed_description`.
  - Combines field scores with configurable weights (default **3,2,2,1** respectively), giving titles more influence while still considering longer text.
  - Returns each candidate with both a total score and the field that contributed most.

- **Best-passage extraction & highlighting** (`search_cli.py`)
  - Finds an informative snippet (prefers Abstract → Claims → Description) and highlights matched terms for a quick evaluation.
  - Optional `--show-passages` shows the best passage from the highest-scoring field.

### Enhancements selected 
1) **Hybrid Searching**
   - Filters supported in both CLI and interactive app are for classification codes, keywords in title or abstract, and specfic titles:
   - Examples:
     - `--class-prefix B60B` (or any prefix, case-insensitive).
     - `--title-kw "spoke,vent"` and `--abstract-kw "cooling,vent"` (AND semantics).
     - `--exact-title "EXACT TITLE"` (case-insensitive exact match).
   - **Why:** Patent agents often search within a specific Cooperative Patent Classification category (e.g., B60B) and use targeted keywords. Hybrid filters help remove irrelevant results and align with how they typically work.

2) **Two-Phase Searching**  
   - Phase-1 BM25 produces a candidate pool; Phase-2 **proximity/phrase reranker** boosts:
     - adjacency (bigrams from the query),
     - local coverage of query terms inside a sliding window,
     - title proximity (extra weight in titles).
   - Adjustable pool size via `--phase1-k` (default 100).
   - **Why:** Improves ranking quality for the very top results, which matter most in practice.

3. **Interfaces & Users**
   - **Terminal interface** (`app.py`): users log in with a username, run searches, and view their **search history**.
     - History viewer lets you choose how many recent searches to show (**default 10**).
     - Commands: `help`, `history [n]`, and options like `class=...`, `titlekw=...`, `abskw=...`, `two=1`, `p1k=...`, `top=...`.
   - **Multi-user support:** open multiple terminals to use the app simultaneously; each user’s history is saved under `~/.thinkstruct`.
   - **Concurrent simulation** (`simulate_users.py`): load-tests parallel use.
     - You can **choose how many users to simulate** with `--users N` (e.g., `--users 100`), and optionally enable two-phase with `--two-phase` and set the pool via `--phase1-k`.
     - Example:
       ```bash
       python3 simulate_users.py --data-dir ./patent_data_small --users 100 --two-phase --phase1-k 100
       ```

---

## How to run the code

### Requirements
- **Python 3.9+** installed on your system (uses only the Python standard library; no external dependencies).
- After unzipping the project folder, ensure the included **`patent_data_small/`** directory (already in the zip) remains in the same location as the Python files.
- No installation or setup is required, simply run the scripts directly from the unzipped folder.

### Running the Command-line search (quick start)
```bash
# Basic search
python3 search_cli.py --data-dir ./patent_data_small \
  --query "wheel assembly"

# Show top 5 with timings and the best passage from the best field
python3 search_cli.py --data-dir ./patent_data_small \
  --query "multi-piece wheel frame" --top-k 5 --time --show-passages

# Hybrid filters: classfication code prefix + title keywords + abstract keywords
python3 search_cli.py --data-dir ./patent_data_small \
  --query "swivel castor" \
  --class-prefix B60B \
  --title-kw "spoke,vent" \
  --abstract-kw "cooling"

# Exact title match (case-insensitive)
python3 search_cli.py --data-dir ./patent_data_small \
  --query "spindle repair apparatus" \
  --exact-title "SPINDLE REPAIR APPARATUS AND METHOD"

# Two-phase rerank on top-100 candidates (then return top-5)
python3 search_cli.py --data-dir ./patent_data_small \
  --query "braking assembly segmented disc" \
  --two-phase --phase1-k 100 --top-k 5 --time

# JSON (NDJSON) output (one line per result) for scripting
python3 search_cli.py --data-dir ./patent_data_small \
  --query "hub flange" --json

# Optional: change field weights (title,abstract,claims,description)
python3 search_cli.py --data-dir ./patent_data_small \
  --query "motorcycle wheel adapter" --weights 3,2,2,1
```

### Running the Interactive App (`app.py`)

The interactive app provides a terminal-based search interface with per-user sessions and search history.

#### 1. Create and activate a virtual environment

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

---

#### 2. Run the app
```bash
python3 app.py
```

---

#### 3. When prompted:
- **Data directory**: enter the path to the patent data (e.g., `./patent_data_small`)
- **Username**: choose any username to save and access your search history
- You can then:
  - Run searches by typing your query and optional parameters (`class=...`, `titlekw=...`, `abskw=...`, `two=1`, `p1k=...`, `top=...`)
  - View history with `history` or `history [n]` (default is last 10)
  - Type `help` for available commands  
  - Type `quit` to exit

---

#### 4. Multi-user support
- You can open multiple terminal windows, run the app in each, and log in as different users.
- Each user’s history is saved under `~/.thinkstruct`.

## Testing Features

### 1) CLI — baseline (no hybrid)
```bash
# Basic search (no filters) — show timing
python3 search_cli.py --data-dir ./patent_data_small \
  --query "wheel assembly" --top-k 5 --time
```

### 2) CLI — hybrid search (single filter)
```bash
# Add a Cooperative Patent Classification prefix filter — compare timing vs baseline
python3 search_cli.py --data-dir ./patent_data_small \
  --query "wheel assembly" \
  --class-prefix B60B \
  --top-k 5 --time
```

### 3) CLI — hybrid search (multiple filters)
```bash
# Combine CPC prefix + title AND abstract keyword filters — still show timing
python3 search_cli.py --data-dir ./patent_data_small \
  --query "wheel assembly" \
  --class-prefix B60B \
  --abstract-kw "cooling" \
  --top-k 5 --time
```

### 4) CLI — hybrid search (exact title)
```bash
# Exact title search within hybrid filters
python3 search_cli.py --data-dir ./patent_data_small \
  --query "spindle repair apparatus" \
  --exact-title "SPINDLE REPAIR APPARATUS AND METHOD" \
  --class-prefix B60B \
  --top-k 5 --time
```

### 5) CLI — two-phase searching (no filters)
```bash
# Two-phase: BM25 → proximity/phrase rerank on top-100, then return top-5
python3 search_cli.py --data-dir ./patent_data_small \
  --query "braking assembly segmented disc" \
  --two-phase --phase1-k 100 \
  --top-k 5 --time --show-passages
```

### 6) CLI — two-phase searching (with filters)
```bash
# Two-phase + hybrid filters combined
python3 search_cli.py --data-dir ./patent_data_small \
  --query "swivel castor" \
  --class-prefix B60B \
  --title-kw "spoke,vent" \
  --abstract-kw "cooling" \
  --two-phase --phase1-k 100 \
  --top-k 5 --time
```

---

### 7) Interactive app — run the same flows

**Start the app**
```bash
python3 app.py
```

**When prompted**
- Data directory: `./patent_data_small`
- Username: pick any (creates per-user history)

**In the app, type these lines (press Enter after each):**
```
"wheel assembly" top=5
"wheel assembly" class=B60B top=5
"swivel castor" class=B60B titlekw=spoke,vent abskw=cooling top=5
"spindle repair apparatus" class=B60B title="SPINDLE REPAIR APPARATUS AND METHOD" top=5
"braking assembly segmented disc" two=1 p1k=100 top=5
"swivel castor" class=B60B titlekw=spoke,vent abskw=cooling two=1 p1k=100 top=5
history
quit
```

---

### 8) Concurrent users — simulate 100 users
```bash
# Load test with 100 parallel users (enable two-phase to stress rerank path)
python3 simulate_users.py --data-dir ./patent_data_small \
  --users 100 --two-phase --phase1-k 100
```