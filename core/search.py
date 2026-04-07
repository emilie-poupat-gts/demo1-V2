import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# LEXICAL SEARCH
# ---------------------------------------------------------
def lexical_search(df, query):
    """Simple keyword-based search."""
    query = query.lower()
    mask = (
        df["title"].str.contains(query, case=False, na=False) |
        df["description"].str.contains(query, case=False, na=False) |
        df["keywords"].str.contains(query, case=False, na=False) |
        df["category"].str.contains(query, case=False, na=False)
    )
    return df[mask]


# ---------------------------------------------------------
# COMBINED SEARCH
# ---------------------------------------------------------
def combined_search(df, categories, years, keywords):
    """Filter by category, year range, and keywords."""
    results = df.copy()

    if categories:
        results = results[results["category"].isin(categories)]

    if years:
        min_year, max_year = years
        results = results[(results["year"] >= min_year) & (results["year"] <= max_year)]

    if keywords:
        for kw in keywords:
            results = results[results["keywords"].str.contains(kw, case=False, na=False)]

    return results


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------
def semantic_search(df, query, model, embeddings_db, top_k=10):
    """Semantic search using sentence-transformer embeddings."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings_db)[0]

    df["similarity"] = similarities
    results = df.sort_values(by="similarity", ascending=False).head(top_k)

    return results.drop(columns=["similarity"])

# -----------------------------------------------------
# SEMANTIC SEARCH WITH YEAR RANGE
# ----------------------------------------------------
def parse_year_filter(query):
    """
    Returns (min_year, max_year, reason) where min_year or max_year can be None.
    Semantics:
      - "from 2000", "since 2000" -> min_year = 2000 (inclusive)
      - "after 2000" -> min_year = 2001 (strictly after)
      - "before 2000", "until 2000", "up to 2000" -> max_year = 1999 (strictly before) or <=2000 depending on phrase
      - "<=2000" or "up to 2000" -> max_year = 2000 (inclusive)
      - "between 2000 and 2005", "2000-2005", "2000 to 2005" -> inclusive range
      - "2000s" -> 2000..2009
      - "in 2000", "movie 2000" -> exact year (min=max=2000)
    """
    q = query.lower()

    # decade "2000s"
    m = re.search(r"\b(19|20)\d{2}s\b", q)
    if m:
        start = int(m.group(0)[:4])
        return start, start + 9, "decade"

    # explicit range "2000-2005" or "2000 to 2005" or "between 2000 and 2005"
    m = re.search(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return int(m.group(1)), int(m.group(2)), "range-dash"
    m = re.search(r"\bbetween\s+(19\d{2}|20\d{2})\s+(?:and|-|to)\s+(19\d{2}|20\d{2})\b", q)
    if m:
        return int(m.group(1)), int(m.group(2)), "between"

    # "from 2000", "since 2000" -> inclusive
    m = re.search(r"\b(?:from|since)\s+(19\d{2}|20\d{2})\b", q)
    if m:
        return int(m.group(1)), None, "from/since"

    # "after 2000" -> strictly greater
    m = re.search(r"\bafter\s+(19\d{2}|20\d{2})\b", q)
    if m:
        return int(m.group(1)) + 1, None, "after"

    # "before 2000" -> strictly less
    m = re.search(r"\bbefore\s+(19\d{2}|20\d{2})\b", q)
    if m:
        return None, int(m.group(1)) - 1, "before"

    # "until 2000" or "up to 2000" -> inclusive <=2000
    m = re.search(r"\b(?:until|up to|<=)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return None, int(m.group(1)), "until/up to"

    # comparison operators like >=2000, >2000, <=2000, <2000
    m = re.search(r"\b(>=|<=|>|<)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        op, y = m.group(1), int(m.group(2))
        if op == ">":
            return y + 1, None, ">"
        if op == ">=":
            return y, None, ">="
        if op == "<":
            return None, y - 1, "<"
        return None, y, "<="

    # "in 2000" or single year with context words -> treat as exact year
    m = re.search(r"\bin\s+(19\d{2}|20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return y, y, "in"

    # multiple years or single year tokens
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", q)
    if len(years) == 1:
        # if query contains "from" treat as from-year
        if re.search(r"\bfrom\b", q):
            return int(years[0]), None, "single-from"
        # default: exact match
        y = int(years[0])
        return y, y, "single-exact"
    if len(years) > 1:
        ys = sorted(int(y) for y in years)
        return ys[0], ys[-1], "multiple-years"

    return None, None, None

def semantic_search_with_year_range(df, query, model, embeddings_db, top_k=10, fallback_to_full=True):
    min_year, max_year, reason = parse_year_filter(query)

    if min_year is not None or max_year is not None:
        # build inclusive mask
        if min_year is None:
            mask = df["year"] <= max_year
        elif max_year is None:
            mask = df["year"] >= min_year
        else:
            mask = (df["year"] >= min_year) & (df["year"] <= max_year)

        if not mask.any():
            if fallback_to_full:
                st.warning(f"No items match the year filter ({reason}). Falling back to full search.")
                candidate_df = df.reset_index(drop=True)
                candidate_embeddings = embeddings_db
            else:
                return pd.DataFrame(columns=list(df.columns) + ["score"])
        else:
            candidate_idx = np.where(mask)[0]
            candidate_df = df.iloc[candidate_idx].reset_index(drop=True)
            candidate_embeddings = embeddings_db[candidate_idx]
    else:
        candidate_df = df.reset_index(drop=True)
        candidate_embeddings = embeddings_db

    q_emb = model.encode(query, convert_to_numpy=True)
    sims = cosine_similarity([q_emb], candidate_embeddings)[0]
    top_local = np.argsort(sims)[::-1][:top_k]
    results = candidate_df.iloc[top_local].copy()
    results["score"] = sims[top_local]
    return results.reset_index(drop=True)