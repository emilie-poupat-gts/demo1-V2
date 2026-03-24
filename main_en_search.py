import streamlit as st
import pandas as pd

from core.text_extractor import extract_text
from core.llm_analyzer import analyze_with_llm
from core.ml_classifier import classify
from core.database import load_database, add_entry
from core.search import (
    lexical_search,
    combined_search,
    semantic_search
)
from core.config import llm, clf, vectorizer


# ---------------------------------------------------------
# INTERFACE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Document tool",
    layout="wide"
)

st.title("Document tool — contribution & search")
st.markdown("RETEX Project — POC1 — Phase 1")


# ---------------------------------------------------------
# LOAD DATABASE
# ---------------------------------------------------------
df = load_database()

st.sidebar.header("Main menu")
mode = st.sidebar.radio(
    "Choose a mode:",
    [
        " Show database",
        " Lexical search",
        " Combined search (with filters)",
        " Semantic search"
    ]
)


# ---------------------------------------------------------
# RESULT DISPLAY FUNCTION
# ---------------------------------------------------------
def display_results(results):
    if results.empty:
        st.warning("No results found")
        return

    st.success(f"{len(results)} result(s) found")

    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['year']}) — {row['category']}"):
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Keywords:** {row['keywords']}")



# ---------------------------------------------------------
# MODE 2 — DATABASE
# ---------------------------------------------------------
if mode == " Show database":
    st.header("Database")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# MODE 3 — LEXICAL SEARCH
# ---------------------------------------------------------
elif mode == " Lexical search":
    st.header("Lexical search")

    query = st.text_input("Type a word or sentence")

    if query:
        results = lexical_search(df, query)
        display_results(results)


# ---------------------------------------------------------
# MODE 4 — COMBINED SEARCH
# ---------------------------------------------------------
elif mode == " Combined search (with filters)":
    st.header("Combined search")

    categories = sorted(df["category"].unique())
    keywords = sorted(list(set(",".join(df["keywords"]).split(", "))))
    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    col1, col2 = st.columns(2)

    with col1:
        selected_categories = st.multiselect("Categories:", categories)

    with col2:
        selected_years = st.slider("Years:", year_min, year_max, (year_min, year_max))

    selected_keywords = st.multiselect("Keywords:", keywords)

    if st.button("Filter"):
        results = combined_search(df, selected_categories, selected_years, selected_keywords)
        display_results(results)


# ---------------------------------------------------------
# MODE 5 — SEMANTIC SEARCH
# ---------------------------------------------------------
elif mode == " Semantic search":
    st.header("Semantic search")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("./modelHF")
    embeddings_db = model.encode(df["description"].tolist())

    query = st.text_input("Describe what you are looking for")

    if query:
        results = semantic_search(df, query, model, embeddings_db)
        display_results(results)
