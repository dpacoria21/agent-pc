"""Local Python version of the CP RAG student-profile prototype.

The module keeps the experimental objects and functions for Page Nodes,
semantic search, keyword search, hybrid retrieval, simulated student sessions,
profile calculation, recommendation, and retrieval metrics.
"""

import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
class Markdown(str):
    pass

def display(*args, **kwargs):
    return None
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 140)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("Entorno listo.")


# %%
CODEFORCES_API_URL = "https://codeforces.com/api/problemset.problems"

def normalize_tags(tags):
    if not isinstance(tags, list):
        return []
    return list(dict.fromkeys(str(tag).lower().strip() for tag in tags if str(tag).strip()))

def estimate_n_limit(rating):
    if pd.isna(rating):
        return 2000
    rating = int(rating)
    if rating <= 1000:
        return 2000
    if rating <= 1400:
        return 200_000
    if rating <= 1800:
        return 300_000
    return 1_000_000

def build_mock_sections(name, rating, tags):
    tags = normalize_tags(tags)
    tag_text = ", ".join(tags) if tags else "general reasoning"
    n_limit = estimate_n_limit(rating)

    statement_bits = [
        f"Problem '{name}' asks the contestant to transform an input into an optimal answer.",
        f"The intended topic signals are: {tag_text}.",
        f"The rating is around {rating}, so the solution should balance correctness proof and implementation detail.",
    ]

    observation_bits = [
        "Look for a small invariant, a monotonic property, or a decomposition of the input.",
        "Compare brute force with the expected complexity before committing to code.",
    ]

    algorithm_bits = [
        "Start with parsing and normalization of the input.",
        "Derive the main state or ordering rule, then implement it with careful boundary checks.",
    ]

    mistakes_bits = [
        "Common mistakes include missing n=1, assuming sorted input, overflow, and off-by-one boundaries.",
        "If the complexity is too high, the solution may receive TLE even when the idea seems correct.",
    ]

    if "greedy" in tags or "sortings" in tags:
        observation_bits.append("For greedy solutions, try to prove that always choosing the locally best candidate is safe.")
        observation_bits.append("An exchange argument or invariant often explains why sorting does not lose optimality.")
        algorithm_bits.append("Sort candidates, scan once, choose the best feasible option, and maintain the invariant.")
        mistakes_bits.append("A frequent WA is using a greedy rule without proving the exchange step or tie handling.")

    if "dp" in tags:
        statement_bits.append("The input likely contains overlapping subproblems where local choices interact.")
        observation_bits.append("Define a DP state that captures exactly the information needed for future transitions.")
        algorithm_bits.append("Initialize base cases, compute transitions, and check memory/time constraints.")
        mistakes_bits.append("Typical DP errors are wrong state definition, missing transition cases, and invalid base cases.")

    if "graphs" in tags or "dfs and similar" in tags or "trees" in tags:
        observation_bits.append("Model the objects as graph nodes and edges; connectivity or traversal order may be central.")
        algorithm_bits.append("Use BFS/DFS, tree DP, DSU, or shortest paths depending on the graph structure.")
        mistakes_bits.append("Graph bugs often come from revisiting nodes, disconnected components, or recursion depth.")

    if "data structures" in tags or "dsu" in tags:
        observation_bits.append("The key may be maintaining dynamic information under updates or queries.")
        algorithm_bits.append("Consider Fenwick tree, segment tree, priority queue, DSU, or ordered sets.")
        mistakes_bits.append("Implementation mistakes include wrong indexing, lazy propagation bugs, and stale values.")

    if "math" in tags or "number theory" in tags or "combinatorics" in tags:
        observation_bits.append("Search for a formula, parity argument, divisibility condition, or counting invariant.")
        algorithm_bits.append("Turn the mathematical observation into direct computation with modular arithmetic if needed.")
        mistakes_bits.append("Math solutions often fail on overflow, modulo normalization, or untested corner cases.")

    if "brute force" in tags:
        observation_bits.append("A constrained brute force may pass if pruning or enumeration bounds are tight.")
        algorithm_bits.append("Try all candidates only after proving the search space is small enough.")
        mistakes_bits.append("Blind try-all solutions are risky when O(n^2) or worse exceeds the constraints.")

    if "implementation" in tags:
        observation_bits.append("The challenge may be translating many conditions into clean branches.")
        algorithm_bits.append("Keep helper functions small, name intermediate cases, and test sample-like edge cases.")
        mistakes_bits.append("Implementation-heavy tasks often fail due to branch order, indexing, or forgotten corner cases.")

    return {
        "statement_mock": " ".join(statement_bits),
        "constraints_mock": (
            f"Assume 1 <= n <= {n_limit}. Multiple test cases may appear. "
            f"A solution near O(n log n) is usually safe at rating {rating}; O(n^2) may be too slow for large n."
        ),
        "examples_mock": (
            "Example mock: for a tiny input, manually trace n=1, already optimal data, reversed data, and duplicated values. "
            "The expected output follows the invariant described in the observation section."
        ),
        "editorial_observation_mock": " ".join(observation_bits),
        "editorial_algorithm_mock": " ".join(algorithm_bits),
        "common_mistakes_mock": " ".join(mistakes_bits),
    }

MOCK_PROBLEM_TEMPLATES = [
    (1001, "A", "Greedy Playlist", 900, ["greedy", "sortings"], 18430),
    (1002, "B", "Balanced Teams", 1200, ["greedy", "implementation"], 12880),
    (1003, "C", "State Compression Walk", 1700, ["dp", "bitmasks"], 4380),
    (1004, "D", "Two Arrays Transition", 1600, ["dp", "implementation"], 5120),
    (1005, "E", "Exchange Argument", 1500, ["greedy", "math"], 6200),
    (1006, "F", "Corner Case Counter", 1100, ["implementation", "brute force"], 15300),
    (1007, "G", "Tree Distances", 1800, ["graphs", "trees", "dfs and similar"], 3900),
    (1008, "H", "Fenwick Inversions", 1900, ["data structures", "sortings"], 2760),
    (1009, "A", "Modulo Game", 1300, ["math", "number theory"], 9900),
    (1010, "B", "Proof of Sorting", 1400, ["greedy", "sortings"], 8300),
    (1011, "C", "Knapsack Signals", 1800, ["dp"], 3600),
    (1012, "D", "Debug the Simulator", 1000, ["implementation"], 20100),
    (1013, "E", "Components Online", 1700, ["graphs", "dsu", "data structures"], 3300),
    (1014, "F", "Counting Paths", 2000, ["dp", "graphs"], 2200),
    (1015, "G", "Parity Proof", 1500, ["math", "greedy"], 5700),
    (1016, "H", "Try All Masks", 1600, ["brute force", "bitmasks"], 4400),
    (1017, "A", "Segment Tree Hotel", 2100, ["data structures"], 1800),
    (1018, "B", "Shortest Escape", 1500, ["graphs", "bfs"], 6500),
    (1019, "C", "DP With Prefixes", 1700, ["dp", "prefix sums"], 4100),
    (1020, "D", "Many Ifs", 900, ["implementation"], 23000),
    (1021, "E", "Combinatorics Table", 1900, ["math", "combinatorics", "dp"], 2600),
    (1022, "F", "Greedy Fails", 1600, ["greedy", "constructive algorithms"], 4700),
    (1023, "G", "Edge Case Marathon", 1200, ["implementation", "math"], 11400),
    (1024, "H", "DSU Queries", 1800, ["dsu", "data structures", "graphs"], 3100),
]

def build_mock_problem_dataset():
    rows = []
    for contest_id, index, name, rating, tags, solved_count in MOCK_PROBLEM_TEMPLATES:
        sections = build_mock_sections(name, rating, tags)
        rows.append({
            "problem_id": f"{contest_id}{index}",
            "contestId": contest_id,
            "index": index,
            "name": name,
            "rating": rating,
            "tags": normalize_tags(tags),
            "solvedCount": solved_count,
            **sections,
        })
    return pd.DataFrame(rows)

def fetch_codeforces_problems(max_problems=80, min_rating=800, max_rating=2400, timeout=12):
    try:
        response = requests.get(CODEFORCES_API_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            raise RuntimeError(f"Codeforces status: {payload.get('status')}")

        result = payload["result"]
        stats_map = {
            (item.get("contestId"), item.get("index")): item.get("solvedCount", 0)
            for item in result.get("problemStatistics", [])
        }

        rows = []
        for problem in result.get("problems", []):
            contest_id = problem.get("contestId")
            index = problem.get("index")
            rating = problem.get("rating")
            name = problem.get("name", "Unnamed problem")
            tags = normalize_tags(problem.get("tags", []))

            if contest_id is None or index is None or rating is None:
                continue
            if not (min_rating <= int(rating) <= max_rating):
                continue

            solved_count = stats_map.get((contest_id, index), 0)
            sections = build_mock_sections(name, rating, tags)
            rows.append({
                "problem_id": f"{contest_id}{index}",
                "contestId": contest_id,
                "index": index,
                "name": name,
                "rating": int(rating),
                "tags": tags,
                "solvedCount": int(solved_count),
                **sections,
            })

        if not rows:
            raise RuntimeError("La API respondió, pero no quedaron problemas tras filtrar.")

        rng = random.Random(RANDOM_SEED)
        rng.shuffle(rows)
        print(f"Codeforces cargó {len(rows)} problemas candidatos. Usando {min(max_problems, len(rows))}.")
        return pd.DataFrame(rows[:max_problems])

    except Exception as exc:
        print("No se pudo usar Codeforces; se activa dataset mock local.")
        print("Detalle:", repr(exc))
        return None

def get_problem_dataset(use_codeforces=True, max_problems=80):
    if use_codeforces:
        df = fetch_codeforces_problems(max_problems=max_problems)
        if df is not None and not df.empty:
            return df
    return build_mock_problem_dataset()

problems_df = get_problem_dataset(use_codeforces=False, max_problems=80)
problems_df["tags_text"] = problems_df["tags"].apply(lambda tags: ", ".join(tags))

display(Markdown(f"**Problemas cargados:** {len(problems_df)}"))
display(problems_df[[
    "problem_id", "contestId", "index", "name", "rating", "tags", "solvedCount"
]].head(12))

tag_counts = Counter(tag for tags in problems_df["tags"] for tag in tags)
display(pd.DataFrame(tag_counts.most_common(15), columns=["tag", "count"]))

problems_df["rating"].hist(bins=12)
plt.title("Distribución de rating de problemas")
plt.xlabel("rating")
plt.ylabel("cantidad")
plt.close('all')


# %%
NODE_SPECS = [
    ("STATEMENT", "Problem statement", "statement_mock"),
    ("CONSTRAINTS", "Constraints", "constraints_mock"),
    ("EXAMPLES", "Examples", "examples_mock"),
    ("EDITORIAL_OBSERVATION", "Editorial observation", "editorial_observation_mock"),
    ("EDITORIAL_ALGORITHM", "Editorial algorithm", "editorial_algorithm_mock"),
    ("COMMON_MISTAKES", "Common mistakes", "common_mistakes_mock"),
]

def create_page_nodes(problems):
    rows = []
    for _, problem in tqdm(problems.iterrows(), total=len(problems), desc="Creando nodos"):
        for node_type, node_title, source_col in NODE_SPECS:
            rows.append({
                "node_id": f"{problem['problem_id']}::{node_type}",
                "problem_id": problem["problem_id"],
                "node_type": node_type,
                "node_title": f"{problem['name']} - {node_title}",
                "node_text": problem[source_col],
                "rating": int(problem["rating"]) if not pd.isna(problem["rating"]) else np.nan,
                "tags": problem["tags"],
                "solvedCount": int(problem["solvedCount"]),
            })
    return pd.DataFrame(rows)

page_nodes_df = create_page_nodes(problems_df)

display(Markdown(f"**Nodos indexados:** {len(page_nodes_df)}"))
display(page_nodes_df.head(10))
display(page_nodes_df["node_type"].value_counts().rename_axis("node_type").reset_index(name="count"))


# %%
MODEL_NAME = "all-MiniLM-L6-v2"
node_texts = page_nodes_df["node_text"].fillna("").tolist()

SEMANTIC_BACKEND = None
semantic_model = None
semantic_fallback_vectorizer = None
semantic_svd = None
node_embeddings = None

try:
    if os.environ.get("RAG_FORCE_SEMANTIC_FALLBACK") == "1":
        raise ImportError("RAG_FORCE_SEMANTIC_FALLBACK=1")
    from sentence_transformers import SentenceTransformer

    semantic_model = SentenceTransformer(MODEL_NAME)
    node_embeddings = semantic_model.encode(
        node_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    SEMANTIC_BACKEND = "sentence-transformers"

except Exception as exc:
    print("No se pudo cargar SentenceTransformer; usando fallback TF-IDF + SVD.")
    print("Detalle:", repr(exc))
    semantic_fallback_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    semantic_tfidf = semantic_fallback_vectorizer.fit_transform(node_texts)
    n_components = min(128, max(2, min(semantic_tfidf.shape) - 1))
    semantic_svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    node_embeddings = semantic_svd.fit_transform(semantic_tfidf)
    node_embeddings = normalize(node_embeddings)
    SEMANTIC_BACKEND = "tfidf-svd-fallback"

print("Backend semántico:", SEMANTIC_BACKEND)
print("Shape de embeddings:", node_embeddings.shape)

def _clip_cosine(scores):
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

def encode_query_semantic(query):
    if SEMANTIC_BACKEND == "sentence-transformers":
        query_embedding = semantic_model.encode([query], normalize_embeddings=True)
        return np.asarray(query_embedding[0], dtype=np.float32)

    query_tfidf = semantic_fallback_vectorizer.transform([query])
    query_embedding = semantic_svd.transform(query_tfidf)
    query_embedding = normalize(query_embedding)
    return query_embedding[0]

def semantic_scores_for_query(query):
    query_embedding = encode_query_semantic(query)
    scores = np.dot(node_embeddings, query_embedding)
    return _clip_cosine(scores)

def semantic_search(query, top_k=10):
    scores = semantic_scores_for_query(query)
    results = page_nodes_df.copy()
    results["semantic_score"] = scores
    results = results.sort_values("semantic_score", ascending=False).head(top_k)
    return results[[
        "node_id", "problem_id", "node_type", "node_text",
        "semantic_score", "rating", "tags"
    ]].reset_index(drop=True)

semantic_demo = semantic_search("greedy proof edge cases", top_k=5)
display(semantic_demo)


# %%
def make_keyword_document(row):
    tags_text = " ".join(row["tags"]) if isinstance(row["tags"], list) else ""
    return f"{row['node_title']} {row['node_text']} {tags_text}"

keyword_documents = page_nodes_df.apply(make_keyword_document, axis=1).tolist()

keyword_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
)
keyword_matrix = keyword_vectorizer.fit_transform(keyword_documents)

def keyword_scores_for_query(query):
    query_vec = keyword_vectorizer.transform([query])
    scores = keyword_matrix @ query_vec.T
    return np.asarray(scores.toarray()).ravel()

def keyword_search(query, top_k=10):
    scores = keyword_scores_for_query(query)
    results = page_nodes_df.copy()
    results["keyword_score"] = scores
    results = results.sort_values("keyword_score", ascending=False).head(top_k)
    return results[[
        "node_id", "problem_id", "node_type", "node_text",
        "keyword_score", "rating", "tags"
    ]].reset_index(drop=True)

keyword_demo = keyword_search("dp state transition", top_k=5)
display(keyword_demo)


# %%
def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def query_tokens(query):
    return set(re.findall(r"[a-zA-Z0-9_+]+", normalize_text(query)))

def tag_matches_query(tags, query):
    q = normalize_text(query)
    tokens = query_tokens(query)
    matches = []
    for tag in normalize_tags(tags):
        tag_norm = normalize_text(tag)
        tag_parts = set(tag_norm.split())
        if tag_norm in q or tag_parts.intersection(tokens):
            matches.append(tag)
    return matches

def infer_node_type_preferences(query):
    q = normalize_text(query)
    preferences = set()
    if any(word in q for word in ["prove", "proof", "invariant", "exchange", "why"]):
        preferences.update(["EDITORIAL_OBSERVATION", "EDITORIAL_ALGORITHM"])
    if any(word in q for word in ["mistake", "wa", "wrong", "edge", "corner", "bug", "debug"]):
        preferences.add("COMMON_MISTAKES")
    if any(word in q for word in ["transition", "state", "algorithm", "complexity", "tle"]):
        preferences.update(["EDITORIAL_ALGORITHM", "CONSTRAINTS"])
    if any(word in q for word in ["example", "sample", "trace"]):
        preferences.add("EXAMPLES")
    if any(word in q for word in ["understand", "statement", "input"]):
        preferences.add("STATEMENT")
    return preferences

def metadata_bonus_for_row(row, query, filters=None):
    filters = filters or {}
    bonus = 0.0

    query_tag_hits = tag_matches_query(row["tags"], query)
    bonus += min(0.06, 0.02 * len(query_tag_hits))

    filter_tags = normalize_tags(filters.get("tags", []))
    if filter_tags:
        overlap = set(normalize_tags(row["tags"])).intersection(filter_tags)
        bonus += min(0.05, 0.025 * len(overlap))

    min_rating = filters.get("min_rating")
    max_rating = filters.get("max_rating")
    if min_rating is not None or max_rating is not None:
        rating = row["rating"]
        lower_ok = True if min_rating is None else rating >= min_rating
        upper_ok = True if max_rating is None else rating <= max_rating
        if lower_ok and upper_ok:
            bonus += 0.03

    preferred_node_types = infer_node_type_preferences(query)
    filter_node_types = set(filters.get("node_types", []))
    if row["node_type"] in preferred_node_types:
        bonus += 0.05
    if filter_node_types and row["node_type"] in filter_node_types:
        bonus += 0.03

    return min(bonus, 0.16)

def apply_retrieval_filters(df, filters=None):
    if not filters:
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if filters.get("min_rating") is not None:
        mask &= df["rating"] >= filters["min_rating"]
    if filters.get("max_rating") is not None:
        mask &= df["rating"] <= filters["max_rating"]
    if filters.get("tags"):
        filter_tags = set(normalize_tags(filters["tags"]))
        mask &= df["tags"].apply(lambda tags: bool(set(normalize_tags(tags)).intersection(filter_tags)))
    if filters.get("node_types"):
        mask &= df["node_type"].isin(filters["node_types"])

    return df.loc[mask].copy()

def hybrid_search(query, top_k=10, alpha=0.65, filters=None):
    semantic_scores = semantic_scores_for_query(query)
    keyword_scores = keyword_scores_for_query(query)

    results = page_nodes_df.copy()
    results["semantic_score"] = semantic_scores
    results["keyword_score"] = keyword_scores
    results["metadata_bonus"] = results.apply(
        lambda row: metadata_bonus_for_row(row, query, filters=filters),
        axis=1,
    )

    results = apply_retrieval_filters(results, filters)
    results["hybrid_score"] = (
        alpha * results["semantic_score"]
        + (1 - alpha) * results["keyword_score"]
        + results["metadata_bonus"]
    )
    results = results.sort_values("hybrid_score", ascending=False).head(top_k)

    return results[[
        "node_id", "problem_id", "node_type", "node_title", "node_text",
        "semantic_score", "keyword_score", "metadata_bonus", "hybrid_score",
        "rating", "tags", "solvedCount"
    ]].reset_index(drop=True)

hybrid_demo_filters = {
    "min_rating": 1200,
    "max_rating": 1800,
    "tags": ["dp", "greedy"],
    "node_types": ["EDITORIAL_OBSERVATION", "COMMON_MISTAKES"],
}

hybrid_demo = hybrid_search(
    "I need help with dp transitions and greedy proof mistakes",
    top_k=8,
    alpha=0.65,
    filters=hybrid_demo_filters,
)

display(hybrid_demo[[
    "node_id", "node_type", "rating", "tags",
    "semantic_score", "keyword_score", "metadata_bonus", "hybrid_score"
]])


# %%
def short_text(text, max_len=180):
    text = normalize_text(text)
    return text[:max_len] + ("..." if len(text) > max_len else "")

def explain_row_reason(row, query):
    reasons = []
    tag_hits = tag_matches_query(row["tags"], query)
    if tag_hits:
        reasons.append(f"coinciden tags de la query: {', '.join(tag_hits)}")
    preferred = infer_node_type_preferences(query)
    if row["node_type"] in preferred:
        reasons.append(f"el tipo de nodo {row['node_type']} coincide con la etapa inferida")
    if row.get("semantic_score", 0) >= 0.45:
        reasons.append("alta similitud semántica")
    if row.get("keyword_score", 0) >= 0.08:
        reasons.append("coincidencia keyword/TF-IDF visible")
    if row.get("metadata_bonus", 0) > 0:
        reasons.append("recibió bonus por metadatos")
    if not reasons:
        reasons.append("apareció por similitud relativa frente al resto del índice")
    return "; ".join(reasons)

def explain_retrieval(query, results_df):
    display(Markdown(f"**Query original:** `{query}`"))

    if results_df is None or results_df.empty:
        display(Markdown("No hay resultados para explicar."))
        return

    results = results_df.copy().head(10)
    if "hybrid_score" not in results.columns:
        if "semantic_score" in results.columns:
            results["hybrid_score"] = results["semantic_score"]
        elif "keyword_score" in results.columns:
            results["hybrid_score"] = results["keyword_score"]
        else:
            results["hybrid_score"] = 0.0
    for column in ["semantic_score", "keyword_score", "metadata_bonus"]:
        if column not in results.columns:
            results[column] = 0.0

    results["snippet"] = results["node_text"].apply(short_text)
    results["why"] = results.apply(lambda row: explain_row_reason(row, query), axis=1)

    display(results[[
        "node_id", "node_type", "rating", "tags",
        "semantic_score", "keyword_score", "metadata_bonus", "hybrid_score",
        "snippet", "why"
    ]])

    plot_df = results.iloc[::-1]
    plt.figure(figsize=(10, 5))
    plt.barh(plot_df["node_id"], plot_df["hybrid_score"])
    plt.title("Top resultados por hybrid_score")
    plt.xlabel("hybrid_score")
    plt.ylabel("node_id")
    plt.tight_layout()
    plt.close('all')

    x = np.arange(len(results))
    width = 0.38
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, results["semantic_score"], width=width, label="semantic_score")
    plt.bar(x + width / 2, results["keyword_score"], width=width, label="keyword_score")
    plt.xticks(x, results["node_id"], rotation=65, ha="right")
    plt.title("Comparación semantic_score vs keyword_score")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.close('all')

explain_retrieval(
    "I cannot prove why sorting greedily is correct and I miss edge cases",
    hybrid_search("I cannot prove why sorting greedily is correct and I miss edge cases", top_k=8),
)


# %%
APPROACH_TO_TAGS = {
    "GREEDY": ["greedy", "sortings", "constructive algorithms"],
    "DP": ["dp", "bitmasks", "prefix sums"],
    "GRAPH": ["graphs", "trees", "dfs and similar", "bfs", "dsu"],
    "DATA_STRUCTURES": ["data structures", "dsu"],
    "MATH": ["math", "number theory", "combinatorics"],
    "BRUTE_FORCE": ["brute force", "bitmasks"],
    "IMPLEMENTATION": ["implementation"],
}

def pick_problem_by_tags(tags, target_rating, offset=0):
    tags = set(normalize_tags(tags))
    candidates = problems_df[
        problems_df["tags"].apply(lambda problem_tags: bool(tags.intersection(normalize_tags(problem_tags))))
    ].copy()
    if candidates.empty:
        candidates = problems_df.copy()
    candidates["rating_distance"] = (candidates["rating"] - target_rating).abs()
    candidates = candidates.sort_values(["rating_distance", "solvedCount"], ascending=[True, False])
    return candidates.iloc[offset % len(candidates)]

def make_idea_text(user_id, approach, solved, risk_type):
    if user_id == "student_A" and approach == "DP":
        return "I think there is a dp state, but I am not sure about the transition and base cases."
    if user_id == "student_A" and approach == "GREEDY":
        return "Sort the elements and always choose the best feasible option; I can use an exchange proof."
    if user_id == "student_B" and risk_type == "WRONG_PROOF":
        return "The implementation seems direct, but I cannot prove why the greedy choice is always valid."
    if user_id == "student_B":
        return "I will implement the cases carefully and test the branches one by one."
    if user_id == "student_C" and risk_type == "EDGE_CASES":
        return "The math formula works generally, but I may be missing edge cases like n=1 or zero values."
    if user_id == "student_C":
        return "Use a parity and modulo observation to derive the formula, then compute it directly."
    if approach == "GRAPH":
        return "Model it as a graph and run bfs or dfs over components."
    return "Try to understand the invariant and implement a clean algorithm."

def simulate_student_data():
    rng = random.Random(RANDOM_SEED)
    base_time = datetime(2026, 1, 10, 18, 0, 0)

    student_specs = {
        "student_A": [
            ("GREEDY", 0.86, "NONE", 1250),
            ("DP", 0.28, "BAD_COMPLEXITY", 1650),
            ("IMPLEMENTATION", 0.68, "EDGE_CASES", 1150),
        ],
        "student_B": [
            ("IMPLEMENTATION", 0.84, "NONE", 1200),
            ("GREEDY", 0.42, "WRONG_PROOF", 1500),
            ("GRAPH", 0.58, "WA", 1550),
        ],
        "student_C": [
            ("MATH", 0.82, "NONE", 1450),
            ("IMPLEMENTATION", 0.35, "EDGE_CASES", 1250),
            ("DP", 0.52, "BAD_COMPLEXITY", 1650),
        ],
    }

    sessions = []
    ideas = []
    attempts = []
    session_counter = 1
    idea_counter = 1
    attempt_counter = 1

    for user_id, specs in student_specs.items():
        for i in range(12):
            approach, solve_prob, default_risk, target_rating = specs[i % len(specs)]
            problem = pick_problem_by_tags(APPROACH_TO_TAGS[approach], target_rating, offset=i)
            solved = rng.random() < solve_prob
            abandoned = (not solved) and (rng.random() < 0.25)

            if abandoned:
                attempts_count = rng.choice([0, 1])
                final_verdict = "ABANDONED"
                first_idea_seconds = rng.randint(480, 1500)
                first_code_seconds = np.nan if attempts_count == 0 else first_idea_seconds + rng.randint(240, 700)
                total_time_seconds = first_idea_seconds + rng.randint(900, 2400)
            elif solved:
                attempts_count = rng.choice([1, 1, 2, 3])
                final_verdict = "AC"
                first_idea_seconds = rng.randint(90, 520) if default_risk == "NONE" else rng.randint(220, 900)
                first_code_seconds = first_idea_seconds + rng.randint(180, 900)
                total_time_seconds = first_code_seconds + rng.randint(600, 2400)
            else:
                attempts_count = rng.choice([2, 3, 4, 5])
                final_verdict = rng.choice(["WA", "TLE", "RE"])
                first_idea_seconds = rng.randint(380, 1400)
                first_code_seconds = first_idea_seconds + rng.randint(300, 1000)
                total_time_seconds = first_code_seconds + rng.randint(1200, 3600)

            session_id = f"s{session_counter:03d}"
            sessions.append({
                "session_id": session_id,
                "user_id": user_id,
                "problem_id": problem["problem_id"],
                "started_at": base_time + timedelta(days=session_counter),
                "first_idea_seconds": int(first_idea_seconds),
                "first_code_seconds": float(first_code_seconds) if not pd.isna(first_code_seconds) else np.nan,
                "total_time_seconds": int(total_time_seconds),
                "attempts_count": int(attempts_count),
                "final_verdict": final_verdict,
                "solved": bool(solved),
                "abandoned": bool(abandoned),
            })

            if default_risk == "NONE" and not solved:
                risk_type = rng.choice(["WA", "EDGE_CASES"])
            else:
                risk_type = default_risk

            idea_quality = "CORRECT" if solved and default_risk == "NONE" else ("PARTIAL" if solved else rng.choice(["PARTIAL", "INCORRECT"]))
            reasoning_stage = "PROOF" if risk_type == "WRONG_PROOF" else ("DEBUGGING" if risk_type in ["WA", "EDGE_CASES"] else "ALGORITHM")
            hint_level = rng.choice([0, 1, 2]) if solved else rng.choice([3, 4, 5])

            ideas.append({
                "idea_id": f"i{idea_counter:03d}",
                "session_id": session_id,
                "user_id": user_id,
                "problem_id": problem["problem_id"],
                "idea_text": make_idea_text(user_id, approach, solved, risk_type),
                "created_second": int(first_idea_seconds),
                "detected_approach": approach,
                "reasoning_stage": reasoning_stage,
                "idea_quality": idea_quality,
                "risk_type": risk_type if risk_type != "WA" else "UNKNOWN",
                "hint_level_requested": int(hint_level),
            })
            idea_counter += 1

            if (not solved) and attempts_count >= 2:
                ideas.append({
                    "idea_id": f"i{idea_counter:03d}",
                    "session_id": session_id,
                    "user_id": user_id,
                    "problem_id": problem["problem_id"],
                    "idea_text": "After WA I should debug corner cases and check whether O(n^2) is too slow.",
                    "created_second": int(min(total_time_seconds - 200, first_code_seconds + 600)),
                    "detected_approach": "IMPLEMENTATION",
                    "reasoning_stage": "DEBUGGING",
                    "idea_quality": "PARTIAL",
                    "risk_type": "EDGE_CASES" if risk_type != "BAD_COMPLEXITY" else "BAD_COMPLEXITY",
                    "hint_level_requested": int(rng.choice([3, 4])),
                })
                idea_counter += 1

            for attempt_no in range(1, attempts_count + 1):
                if solved and attempt_no == attempts_count:
                    verdict = "AC"
                    error_type = "NONE"
                else:
                    if risk_type == "BAD_COMPLEXITY":
                        verdict = rng.choice(["TLE", "WA"])
                        error_type = "BAD_COMPLEXITY"
                    elif risk_type == "WRONG_PROOF":
                        verdict = "WA"
                        error_type = "LOGIC_BUG"
                    elif risk_type == "EDGE_CASES":
                        verdict = rng.choice(["WA", "RE"])
                        error_type = "EDGE_CASE"
                    else:
                        verdict = rng.choice(["WA", "CE", "RE"])
                        error_type = rng.choice(["IMPLEMENTATION_BUG", "SYNTAX", "RUNTIME"])

                submitted_second = 0
                if not pd.isna(first_code_seconds):
                    submitted_second = int(first_code_seconds + attempt_no * max(120, (total_time_seconds - first_code_seconds) / max(1, attempts_count + 1)))

                attempts.append({
                    "attempt_id": f"a{attempt_counter:03d}",
                    "session_id": session_id,
                    "user_id": user_id,
                    "problem_id": problem["problem_id"],
                    "language": rng.choice(["Python 3", "PyPy 3", "C++17"]),
                    "verdict": verdict,
                    "submitted_second": submitted_second,
                    "code_summary": f"{approach.lower()} solution attempt with {attempt_no} submission(s)",
                    "estimated_complexity": rng.choice(["O(n)", "O(n log n)", "O(n^2)", "O(n*m)"]),
                    "error_type": error_type,
                })
                attempt_counter += 1

            session_counter += 1

    return pd.DataFrame(sessions), pd.DataFrame(ideas), pd.DataFrame(attempts)

sessions_df, ideas_df, attempts_df = simulate_student_data()

display(Markdown("**sessions_df**"))
display(sessions_df.head(12))
display(Markdown("**ideas_df**"))
display(ideas_df.head(12))
display(Markdown("**attempts_df**"))
display(attempts_df.head(12))

display(sessions_df.groupby("user_id")[["solved", "abandoned", "attempts_count"]].agg(["mean", "sum", "count"]))


# %%
def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)

def analyze_student_idea(idea_text):
    text = normalize_text(idea_text)

    approach = "UNKNOWN"
    if contains_any(text, ["segment tree", "fenwick", "dsu", "ordered set", "priority queue"]):
        approach = "DATA_STRUCTURES"
    elif contains_any(text, ["dp", "state", "transition", "memo", "base case"]):
        approach = "DP"
    elif contains_any(text, ["graph", "bfs", "dfs", "tree", "component", "shortest path"]):
        approach = "GRAPH"
    elif contains_any(text, ["sort", "greedy", "choose", "always", "exchange"]):
        approach = "GREEDY"
    elif contains_any(text, ["math", "modulo", "parity", "formula", "gcd", "combinatorics"]):
        approach = "MATH"
    elif contains_any(text, ["brute force", "try all", "enumerate all"]):
        approach = "BRUTE_FORCE"
    elif contains_any(text, ["implement", "case", "branch", "parse", "debug"]):
        approach = "IMPLEMENTATION"

    reasoning_stage = "UNDERSTANDING"
    if contains_any(text, ["debug", "wa", "tle", "runtime", "wrong answer"]):
        reasoning_stage = "DEBUGGING"
    elif contains_any(text, ["implement", "code", "branch", "parse"]):
        reasoning_stage = "IMPLEMENTATION"
    elif contains_any(text, ["prove", "proof", "invariant", "exchange"]):
        reasoning_stage = "PROOF"
    elif contains_any(text, ["algorithm", "transition", "bfs", "dfs", "sort", "compute"]):
        reasoning_stage = "ALGORITHM"
    elif contains_any(text, ["maybe", "i think", "hypothesis"]):
        reasoning_stage = "HYPOTHESIS"
    elif contains_any(text, ["observe", "notice", "property"]):
        reasoning_stage = "OBSERVATION"

    risk_type = "UNKNOWN"
    if contains_any(text, ["o(n^2)", "too slow", "tle", "complexity too high"]):
        risk_type = "BAD_COMPLEXITY"
    elif contains_any(text, ["edge", "n=1", "corner", "zero values", "empty"]):
        risk_type = "EDGE_CASES"
    elif contains_any(text, ["prove", "proof", "always valid", "invariant"]):
        risk_type = "WRONG_PROOF"
    elif contains_any(text, ["brute force", "try all"]):
        risk_type = "TLE"
    elif contains_any(text, ["wrong answer", "wa"]):
        risk_type = "WA"
    elif contains_any(text, ["correct", "safe", "works generally"]):
        risk_type = "NONE"

    return {
        "approach": approach,
        "reasoning_stage": reasoning_stage,
        "risk_type": risk_type,
    }

def analyze_student_idea_with_llm(idea_text, provider="openai_or_gemini", api_key=None):
    return {
        "status": "placeholder",
        "provider": provider,
        "message": (
            "Aquí se conectaría un LLM real. El prompt debería pedir JSON estricto con "
            "approach, reasoning_stage, risk_type, confidence y una explicación breve."
        ),
        "input_preview": short_text(idea_text, 120),
    }

heuristic_analysis = ideas_df["idea_text"].apply(analyze_student_idea).apply(pd.Series)
ideas_df = pd.concat(
    [ideas_df, heuristic_analysis.add_prefix("heuristic_")],
    axis=1,
)

display(ideas_df[[
    "idea_id", "user_id", "idea_text",
    "detected_approach", "reasoning_stage", "risk_type",
    "heuristic_approach", "heuristic_reasoning_stage", "heuristic_risk_type"
]].head(12))

display(pd.DataFrame([analyze_student_idea_with_llm("I think sorting greedily works but I cannot prove it.")]))


# %%
def clamp(value, low=0, high=100):
    return max(low, min(high, value))

def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) else 0.0

def get_user_frames(user_id):
    user_sessions = sessions_df[sessions_df["user_id"] == user_id].copy()
    user_ideas = ideas_df[ideas_df["user_id"] == user_id].copy()
    user_attempts = attempts_df[attempts_df["user_id"] == user_id].copy()
    return user_sessions, user_ideas, user_attempts

def tag_performance_for_user(user_sessions):
    joined = user_sessions.merge(problems_df[["problem_id", "name", "rating", "tags"]], on="problem_id", how="left")
    rows = []
    for _, row in joined.iterrows():
        for tag in normalize_tags(row["tags"]):
            rows.append({
                "tag": tag,
                "solved": bool(row["solved"]),
                "attempts_count": int(row["attempts_count"]),
                "rating": int(row["rating"]) if not pd.isna(row["rating"]) else np.nan,
            })
    tag_df = pd.DataFrame(rows)
    if tag_df.empty:
        return pd.DataFrame(columns=["tag", "sessions", "solved_rate", "avg_attempts", "avg_rating"])

    perf = tag_df.groupby("tag").agg(
        sessions=("solved", "count"),
        solved_rate=("solved", "mean"),
        avg_attempts=("attempts_count", "mean"),
        avg_rating=("rating", "mean"),
    ).reset_index()
    return perf.sort_values(["solved_rate", "sessions"], ascending=[False, False])

def calculate_student_profile(user_id):
    user_sessions, user_ideas, user_attempts = get_user_frames(user_id)
    if user_sessions.empty:
        raise ValueError(f"No hay sesiones para user_id={user_id}")

    total_sessions = int(len(user_sessions))
    solved_count = int(user_sessions["solved"].sum())
    abandoned_count = int(user_sessions["abandoned"].sum())
    solved_rate = float(solved_count / total_sessions)
    abandoned_rate = float(abandoned_count / total_sessions)

    solved_sessions = user_sessions[user_sessions["solved"]]
    avg_attempts_to_ac = safe_mean(solved_sessions["attempts_count"])

    dominant_approaches = (
        user_ideas["detected_approach"]
        .value_counts()
        .head(4)
        .to_dict()
    )

    ideas_with_outcome = user_ideas.merge(
        user_sessions[["session_id", "solved"]],
        on="session_id",
        how="left",
    )
    weak_approaches = []
    if not ideas_with_outcome.empty:
        approach_perf = ideas_with_outcome.groupby("detected_approach").agg(
            ideas=("idea_id", "count"),
            solved_rate=("solved", "mean"),
            risk_rate=("risk_type", lambda s: np.mean(~s.isin(["NONE", "UNKNOWN", "WA"]))),
        ).reset_index()
        weak_approaches = approach_perf[
            (approach_perf["ideas"] >= 2)
            & ((approach_perf["solved_rate"] < 0.55) | (approach_perf["risk_rate"] > 0.45))
        ].sort_values(["solved_rate", "risk_rate"], ascending=[True, False])["detected_approach"].tolist()

    common_risks = (
        user_ideas["risk_type"]
        .replace({"WA": "UNKNOWN"})
        .value_counts()
        .drop(labels=["NONE"], errors="ignore")
        .head(5)
        .to_dict()
    )
    common_error_types = (
        user_attempts["error_type"]
        .value_counts()
        .drop(labels=["NONE"], errors="ignore")
        .head(5)
        .to_dict()
    )

    tag_perf = tag_performance_for_user(user_sessions)
    if tag_perf.empty:
        strongest_tags = []
        weakest_tags = []
    else:
        strongest_tags = tag_perf[
            (tag_perf["sessions"] >= 2) & (tag_perf["solved_rate"] >= 0.65)
        ].sort_values(["solved_rate", "avg_attempts"], ascending=[False, True])["tag"].head(5).tolist()
        weakest_tags = tag_perf[
            tag_perf["sessions"] >= 2
        ].sort_values(["solved_rate", "avg_attempts"], ascending=[True, False])["tag"].head(5).tolist()

    high_hint_rate = float((user_ideas["hint_level_requested"] >= 4).mean()) if len(user_ideas) else 0.0
    avg_time_to_first_idea = safe_mean(user_sessions["first_idea_seconds"])
    slow_idea_penalty = clamp((avg_time_to_first_idea - 450) / 700, 0, 1) * 20
    autonomy_score = clamp(100 - 35 * high_hint_rate - 30 * abandoned_rate - slow_idea_penalty)

    proof_ideas = user_ideas[user_ideas["reasoning_stage"] == "PROOF"]
    if len(proof_ideas):
        bad_proof_rate = float(proof_ideas["idea_quality"].isin(["PARTIAL", "INCORRECT"]).mean())
        wrong_proof_rate = float((proof_ideas["risk_type"] == "WRONG_PROOF").mean())
    else:
        bad_proof_rate = 0.25
        wrong_proof_rate = 0.0
    proof_skill_score = clamp(90 - 48 * bad_proof_rate - 22 * wrong_proof_rate)

    impl_error_rate = 0.0
    if len(user_attempts):
        impl_error_rate = float(user_attempts["error_type"].isin([
            "IMPLEMENTATION_BUG", "SYNTAX", "RUNTIME", "EDGE_CASE"
        ]).mean())
    implementation_skill_score = clamp(92 - 55 * impl_error_rate)

    debug_candidates = []
    for session_id, group in user_attempts.groupby("session_id"):
        verdicts = group.sort_values("submitted_second")["verdict"].tolist()
        if any(v in ["WA", "TLE", "RE", "CE"] for v in verdicts):
            debug_candidates.append({
                "session_id": session_id,
                "recovered": verdicts[-1] == "AC" and len(verdicts) <= 3,
            })
    if debug_candidates:
        recovered_rate = float(pd.DataFrame(debug_candidates)["recovered"].mean())
    else:
        recovered_rate = 0.5
    debugging_skill_score = clamp(45 + 55 * recovered_rate - 15 * abandoned_rate)

    profile = {
        "user_id": user_id,
        "total_sessions": total_sessions,
        "solved_count": solved_count,
        "solved_rate": round(solved_rate, 3),
        "abandoned_count": abandoned_count,
        "avg_time_to_first_idea": round(avg_time_to_first_idea, 1),
        "avg_time_to_first_code": round(safe_mean(user_sessions["first_code_seconds"]), 1),
        "avg_total_time": round(safe_mean(user_sessions["total_time_seconds"]), 1),
        "avg_attempts_to_ac": round(avg_attempts_to_ac, 2),
        "dominant_approaches": dominant_approaches,
        "weak_approaches": weak_approaches,
        "common_risks": common_risks,
        "common_error_types": common_error_types,
        "strongest_tags": strongest_tags,
        "weakest_tags": weakest_tags,
        "autonomy_score": round(float(autonomy_score), 1),
        "proof_skill_score": round(float(proof_skill_score), 1),
        "implementation_skill_score": round(float(implementation_skill_score), 1),
        "debugging_skill_score": round(float(debugging_skill_score), 1),
    }
    return profile

def show_student_profile(user_id):
    profile = calculate_student_profile(user_id)
    user_sessions, user_ideas, _ = get_user_frames(user_id)

    display(Markdown(f"## Perfil de `{user_id}`"))
    print(json.dumps(profile, ensure_ascii=False, indent=2))

    summary_rows = [
        ("total_sessions", profile["total_sessions"]),
        ("solved_rate", profile["solved_rate"]),
        ("avg_time_to_first_idea", profile["avg_time_to_first_idea"]),
        ("avg_time_to_first_code", profile["avg_time_to_first_code"]),
        ("avg_total_time", profile["avg_total_time"]),
        ("autonomy_score", profile["autonomy_score"]),
        ("proof_skill_score", profile["proof_skill_score"]),
        ("implementation_skill_score", profile["implementation_skill_score"]),
        ("debugging_skill_score", profile["debugging_skill_score"]),
    ]
    display(pd.DataFrame(summary_rows, columns=["metric", "value"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(["solved", "abandoned"], [profile["solved_count"], profile["abandoned_count"]])
    axes[0, 0].set_title("Solved vs abandoned")
    axes[0, 0].set_ylabel("sesiones")

    approach_counts = user_ideas["detected_approach"].value_counts()
    axes[0, 1].bar(approach_counts.index, approach_counts.values)
    axes[0, 1].set_title("Distribución de enfoques")
    axes[0, 1].tick_params(axis="x", rotation=45)

    risk_counts = user_ideas["risk_type"].value_counts()
    axes[1, 0].bar(risk_counts.index, risk_counts.values)
    axes[1, 0].set_title("Riesgos frecuentes")
    axes[1, 0].tick_params(axis="x", rotation=45)

    time_metrics = {
        "first_idea": profile["avg_time_to_first_idea"],
        "first_code": profile["avg_time_to_first_code"],
        "total": profile["avg_total_time"],
    }
    axes[1, 1].bar(time_metrics.keys(), time_metrics.values())
    axes[1, 1].set_title("Tiempos promedio")
    axes[1, 1].set_ylabel("segundos")

    plt.tight_layout()
    plt.close('all')
    return profile

student_A_profile = show_student_profile("student_A")


# %%
def vector_norm(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

problem_embedding_cache = {}

def get_problem_embedding(problem_id):
    if problem_id in problem_embedding_cache:
        return problem_embedding_cache[problem_id]
    node_indices = page_nodes_df.index[page_nodes_df["problem_id"] == problem_id].tolist()
    if not node_indices:
        emb = np.zeros(node_embeddings.shape[1])
    else:
        emb = vector_norm(node_embeddings[node_indices].mean(axis=0))
    problem_embedding_cache[problem_id] = emb
    return emb

def cosine_between(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def recommended_rating_for_user(user_id):
    user_sessions, _, _ = get_user_frames(user_id)
    solved = user_sessions[user_sessions["solved"]].merge(
        problems_df[["problem_id", "rating"]],
        on="problem_id",
        how="left",
    )
    if solved.empty:
        return 1200
    return int(clamp(float(solved["rating"].median()) + 150, 800, 2400))

def weakness_tags_from_profile(profile):
    weak = set(normalize_tags(profile.get("weakest_tags", [])))
    for approach in profile.get("weak_approaches", []):
        weak.update(APPROACH_TO_TAGS.get(approach, []))
    return weak

def explain_recommendation(row, weak_hits, target_rating):
    pieces = []
    if weak_hits:
        pieces.append(f"trabaja debilidades en {', '.join(sorted(weak_hits))}")
    if abs(row["rating"] - target_rating) <= 250:
        pieces.append("está en un rango de dificultad adecuado")
    if row["semantic_similarity_to_failed_problems"] > 0.35:
        pieces.append("se parece a problemas donde hubo tropiezos previos")
    if row["popularity_score"] > 0.6:
        pieces.append("tiene alta popularidad/resoluciones")
    if not pieces:
        pieces.append("aporta diversidad al plan de práctica")
    return "Recomendado porque " + "; ".join(pieces) + "."

def recommend_problems(user_id, top_k=10):
    profile = calculate_student_profile(user_id)
    user_sessions, _, _ = get_user_frames(user_id)
    solved_problem_ids = set(user_sessions.loc[user_sessions["solved"], "problem_id"])
    failed_problem_ids = set(user_sessions.loc[~user_sessions["solved"], "problem_id"])

    target_rating = recommended_rating_for_user(user_id)
    weak_tags = weakness_tags_from_profile(profile)
    strong_tags = set(normalize_tags(profile.get("strongest_tags", [])))

    if failed_problem_ids:
        failed_embeddings = np.vstack([get_problem_embedding(pid) for pid in failed_problem_ids])
        failed_centroid = vector_norm(failed_embeddings.mean(axis=0))
    else:
        failed_centroid = np.zeros(node_embeddings.shape[1])

    max_popularity = np.log1p(problems_df["solvedCount"]).max()
    rows = []

    for _, problem in problems_df.iterrows():
        if problem["problem_id"] in solved_problem_ids:
            continue

        tags = set(normalize_tags(problem["tags"]))
        weak_hits = tags.intersection(weak_tags)
        weakness_match = min(1.0, len(weak_hits) / max(1, min(3, len(weak_tags) if weak_tags else 1)))

        rating_fit = math.exp(-abs(float(problem["rating"]) - target_rating) / 450)
        semantic_similarity = max(0.0, cosine_between(get_problem_embedding(problem["problem_id"]), failed_centroid))
        popularity_score = float(np.log1p(problem["solvedCount"]) / max_popularity) if max_popularity else 0.0
        diversity_score = 1.0 - min(1.0, len(tags.intersection(strong_tags)) / max(1, len(tags)))

        recommendation_score = (
            0.35 * weakness_match
            + 0.25 * rating_fit
            + 0.20 * semantic_similarity
            + 0.10 * popularity_score
            + 0.10 * diversity_score
        )

        rows.append({
            "problem_id": problem["problem_id"],
            "name": problem["name"],
            "rating": int(problem["rating"]),
            "tags": problem["tags"],
            "weakness_match": round(weakness_match, 3),
            "rating_fit": round(rating_fit, 3),
            "semantic_similarity_to_failed_problems": round(semantic_similarity, 3),
            "popularity_score": round(popularity_score, 3),
            "diversity_score": round(diversity_score, 3),
            "recommendation_score": round(recommendation_score, 3),
            "reason": explain_recommendation({
                "rating": int(problem["rating"]),
                "semantic_similarity_to_failed_problems": semantic_similarity,
                "popularity_score": popularity_score,
            }, weak_hits, target_rating),
        })

    recommendations = pd.DataFrame(rows)
    if recommendations.empty:
        return recommendations
    return recommendations.sort_values("recommendation_score", ascending=False).head(top_k).reset_index(drop=True)

display(recommend_problems("student_A", top_k=10)[[
    "problem_id", "name", "rating", "tags", "recommendation_score", "reason"
]])


# %%
def personalization_bonus_for_row(row, profile):
    bonus = 0.0
    weak_tags = weakness_tags_from_profile(profile)
    row_tags = set(normalize_tags(row["tags"]))
    if weak_tags:
        bonus += min(0.10, 0.04 * len(row_tags.intersection(weak_tags)))

    weak_approaches = set(profile.get("weak_approaches", []))
    if "GREEDY" in weak_approaches and row["node_type"] in ["EDITORIAL_OBSERVATION", "EDITORIAL_ALGORITHM"]:
        bonus += 0.04
    if "IMPLEMENTATION" in weak_approaches and row["node_type"] == "COMMON_MISTAKES":
        bonus += 0.05
    if "DP" in weak_approaches and ("dp" in row_tags or row["node_type"] == "EDITORIAL_ALGORITHM"):
        bonus += 0.05
    return min(bonus, 0.16)

def personalized_hybrid_search(query, user_id, top_k=10, alpha=0.65, filters=None):
    profile = calculate_student_profile(user_id)
    results = hybrid_search(query, top_k=max(top_k * 4, 30), alpha=alpha, filters=filters)
    if results.empty:
        return results
    results["personalization_bonus"] = results.apply(
        lambda row: personalization_bonus_for_row(row, profile),
        axis=1,
    )
    results["personalized_score"] = results["hybrid_score"] + results["personalization_bonus"]
    return results.sort_values("personalized_score", ascending=False).head(top_k).reset_index(drop=True)

def compact_results(df, method_name, score_col):
    out = df.copy().head(8)
    out["method"] = method_name
    out["rank"] = range(1, len(out) + 1)
    out["score"] = out[score_col]
    return out[["method", "rank", "node_id", "problem_id", "node_type", "rating", "tags", "score"]]

def compare_retrieval_methods(query, user_id="student_A", top_k=8):
    sem = semantic_search(query, top_k=top_k)
    key = keyword_search(query, top_k=top_k)
    hyb = hybrid_search(query, top_k=top_k)
    per = personalized_hybrid_search(query, user_id=user_id, top_k=top_k)

    display(Markdown("## Resultados por método"))
    comparison = pd.concat([
        compact_results(sem, "semantic", "semantic_score"),
        compact_results(key, "keyword", "keyword_score"),
        compact_results(hyb, "hybrid", "hybrid_score"),
        compact_results(per, "personalized_hybrid", "personalized_score"),
    ], ignore_index=True)
    display(comparison)

    methods = {
        "semantic": set(sem["node_id"]),
        "keyword": set(key["node_id"]),
        "hybrid": set(hyb["node_id"]),
        "personalized_hybrid": set(per["node_id"]),
    }
    overlap_rows = []
    for left_name, left_set in methods.items():
        row = {"method": left_name}
        for right_name, right_set in methods.items():
            row[right_name] = len(left_set.intersection(right_set))
        overlap_rows.append(row)
    display(Markdown("## Overlap entre métodos por node_id"))
    display(pd.DataFrame(overlap_rows))

    display(Markdown('''
**Lectura rápida:** semantic prioriza significado general; keyword prioriza términos literales; hybrid balancea ambos con metadatos; personalized_hybrid reordena hacia debilidades del estudiante.
'''))
    return {
        "semantic": sem,
        "keyword": key,
        "hybrid": hyb,
        "personalized_hybrid": per,
        "comparison": comparison,
    }

comparison_outputs = compare_retrieval_methods(
    query="I have problems proving greedy solutions and handling edge cases",
    user_id="student_B",
    top_k=8,
)


# %%
def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    return len(set(retrieved_at_k).intersection(relevant_ids)) / len(retrieved_at_k)

def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    return len(set(retrieved_at_k).intersection(relevant_ids)) / len(relevant_ids)

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    for idx, node_id in enumerate(retrieved_ids, start=1):
        if node_id in relevant_ids:
            return 1.0 / idx
    return 0.0

def mean_reciprocal_rank(retrieved_lists, relevant_lists):
    if not retrieved_lists:
        return 0.0
    rr_values = [
        reciprocal_rank(retrieved_ids, relevant_ids)
        for retrieved_ids, relevant_ids in zip(retrieved_lists, relevant_lists)
    ]
    return float(np.mean(rr_values)) if rr_values else 0.0

def average_similarity_score(results_df, score_col="hybrid_score"):
    if results_df.empty or score_col not in results_df.columns:
        return 0.0
    return float(results_df[score_col].mean())

def diversity_by_tags(results_df):
    tags = []
    for row_tags in results_df.get("tags", []):
        tags.extend(normalize_tags(row_tags))
    if not tags:
        return 0.0
    return len(set(tags)) / len(tags)

def coverage_by_node_type(results_df):
    if results_df.empty:
        return 0.0
    total_types = len({spec[0] for spec in NODE_SPECS})
    return results_df["node_type"].nunique() / total_types

def expected_nodes_by_rule(tag_keywords=None, node_types=None, text_keywords=None, top_n=12):
    tag_keywords = set(normalize_tags(tag_keywords or []))
    node_types = set(node_types or [])
    text_keywords = [normalize_text(word) for word in (text_keywords or [])]

    candidates = page_nodes_df.copy()
    mask = pd.Series(True, index=candidates.index)
    if tag_keywords:
        mask &= candidates["tags"].apply(lambda tags: bool(set(normalize_tags(tags)).intersection(tag_keywords)))
    if node_types:
        mask &= candidates["node_type"].isin(node_types)
    if text_keywords:
        mask &= candidates["node_text"].apply(lambda text: any(word in normalize_text(text) for word in text_keywords))

    selected = candidates.loc[mask].sort_values(["solvedCount", "rating"], ascending=[False, True])
    if selected.empty:
        selected = candidates.sort_values(["solvedCount", "rating"], ascending=[False, True])
    return selected["node_id"].head(top_n).tolist()

ground_truth = {
    "greedy proof": expected_nodes_by_rule(
        tag_keywords=["greedy"],
        node_types=["EDITORIAL_OBSERVATION", "EDITORIAL_ALGORITHM", "COMMON_MISTAKES"],
        text_keywords=["proof", "exchange", "greedy"],
        top_n=18,
    ),
    "dp state transition": expected_nodes_by_rule(
        tag_keywords=["dp"],
        node_types=["EDITORIAL_OBSERVATION", "EDITORIAL_ALGORITHM", "COMMON_MISTAKES"],
        text_keywords=["state", "transition", "base cases"],
        top_n=18,
    ),
    "edge cases implementation": expected_nodes_by_rule(
        tag_keywords=["implementation", "brute force", "math"],
        node_types=["COMMON_MISTAKES", "EXAMPLES", "CONSTRAINTS", "EDITORIAL_ALGORITHM"],
        text_keywords=["edge", "corner", "n=1", "implementation"],
        top_n=18,
    ),
}

display(Markdown("**Ground truth simulado**"))
display(pd.DataFrame([
    {"query": query, "expected_node_ids": node_ids}
    for query, node_ids in ground_truth.items()
]))

def evaluate_retrieval(ground_truth, search_fn=hybrid_search, top_k=6):
    rows = []
    reciprocal_ranks = []
    for query, relevant_ids in ground_truth.items():
        results = search_fn(query, top_k=top_k)
        retrieved_ids = results["node_id"].tolist()
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        reciprocal_ranks.append(rr)
        rows.append({
            "query": query,
            "precision_at_k": round(precision_at_k(retrieved_ids, relevant_ids, top_k), 3),
            "recall_at_k": round(recall_at_k(retrieved_ids, relevant_ids, top_k), 3),
            "reciprocal_rank": round(rr, 3),
            "average_similarity_score": round(average_similarity_score(results), 3),
            "diversity_by_tags": round(diversity_by_tags(results), 3),
            "coverage_by_node_type": round(coverage_by_node_type(results), 3),
        })
    evaluation = pd.DataFrame(rows)
    display(Markdown(f"**MRR promedio:** {np.mean(reciprocal_ranks):.3f}"))
    return evaluation

retrieval_evaluation_df = evaluate_retrieval(ground_truth, search_fn=hybrid_search, top_k=6)
display(retrieval_evaluation_df)


# %%
def run_demo(user_id="student_A", query="I think sorting greedily works but I cannot prove it"):
    display(Markdown(f"# Demo RAG adaptativo para `{user_id}`"))

    display(Markdown("## 1. Análisis heurístico de la idea"))
    idea_analysis = analyze_student_idea(query)
    display(pd.DataFrame([{"idea_text": query, **idea_analysis}]))

    display(Markdown("## 2. Perfil actual del estudiante"))
    profile = show_student_profile(user_id)

    display(Markdown("## 3. Contexto recuperado por RAG personalizado"))
    rag_context = personalized_hybrid_search(query, user_id=user_id, top_k=8)
    explain_retrieval(query, rag_context)

    display(Markdown("## 4. Resultados similares"))
    similar_results = semantic_search(query, top_k=8)
    display(similar_results[["node_id", "problem_id", "node_type", "semantic_score", "rating", "tags"]])

    display(Markdown("## 5. Recomendaciones de problemas"))
    recs = recommend_problems(user_id, top_k=8)
    display(recs[["problem_id", "name", "rating", "tags", "recommendation_score", "reason"]])

    display(Markdown("## 6. Métricas de búsqueda"))
    metrics = evaluate_retrieval(ground_truth, search_fn=hybrid_search, top_k=6)
    display(metrics)

    weak_tags = ", ".join(profile.get("weakest_tags", [])) or "sin tags débiles claros"
    weak_approaches = ", ".join(profile.get("weak_approaches", [])) or "sin enfoques débiles claros"
    display(Markdown(f'''
## 7. Explicación final

El sistema detecta la intención cognitiva de la query, recupera nodos específicos del índice y reordena el contexto según el perfil del estudiante.

Para `{user_id}`, las debilidades actuales son enfoques `{weak_approaches}` y tags `{weak_tags}`. Por eso el retrieval personalizado puede elevar observaciones editoriales, errores comunes o algoritmos de problemas relacionados, en vez de devolver solamente statements completos.

Este comportamiento es la base experimental de un tutor RAG: recupera evidencia, explica por qué la recuperó y recomienda práctica ajustada al perfil.
'''))

    return {
        "idea_analysis": idea_analysis,
        "profile": profile,
        "rag_context": rag_context,
        "recommendations": recs,
        "metrics": metrics,
    }

demo_output = run_demo(
    user_id="student_A",
    query="I think sorting greedily works but I cannot prove it",
)
