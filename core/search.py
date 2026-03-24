import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# LEXICAL SEARCH
# ---------------------------------------------------------
def lexical_search(df, query):
    """Simple keyword-based search."""
    mask = (
        df["title"].str.contains(query, case=False, na=False) |
        df["description"].str.contains(query, case=False, na=False) |
        df["keywords"].str.contains(query, case=False, na=False)
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