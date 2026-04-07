import streamlit as st
import pandas as pd

from core.text_extractor import extract_text
from core.llm_analyzer import analyze_with_llm
from core.ml_classifier import classify
from core.database import load_database, add_entry
from core.search import (
    lexical_search,
    combined_search,
    semantic_search,
    semantic_search_with_year_range
)
from core.embedding_helper import (
    load_model,
    compute_embeddings 
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

@st.cache_resource
def load_model(path="./modelHF"):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(path)

# ---------------------------------------------------------
# RESULT DISPLAY FUNCTION
# ---------------------------------------------------------
def display_results_semantic(results):
    if results.empty:
        st.warning("No results found")
        return

    st.success(f"{len(results)} result(s) found")

    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['year']}) — {row['category']} "):
            st.write(f"**Match score:** `{row['score']:.4f}`")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Keywords:** {row['keywords']}")

def display_results_classic(results):
    if results.empty:
        st.warning("No results found")
        return

    st.success(f"{len(results)} result(s) found")

    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['year']}) — {row['category']} "):
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

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    query = st.text_input("Type a word or sentence")

    if query:
        results = lexical_search(df, query)
        display_results_classic(results)


# ---------------------------------------------------------
# MODE 4 — COMBINED SEARCH
# ---------------------------------------------------------
elif mode == " Combined search (with filters)":
    st.header("Combined search")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    categories = sorted(df["category"].unique())
    #keywords = sorted(list(set(",".join(df["keywords"]).split(", "))))
    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    col1, col2 = st.columns(2)

    with col1:
        selected_categories = st.multiselect("Categories:", categories)

    with col2:
        selected_years = st.slider("Years:", year_min, year_max, (year_min, year_max))

    #selected_keywords = st.multiselect("Keywords:", keywords)
    keywords = st.text_input("Select keywords (separeted by a comma) :")
    selected_keywords = sorted(list(set(",".join(keywords).split(","))))

    if st.button("Filter"):
        results = combined_search(df, selected_categories, selected_years, selected_keywords)
        display_results_classic(results)


# ---------------------------------------------------------
# MODE 5 — SEMANTIC SEARCH
# ---------------------------------------------------------
elif mode == " Semantic search":
    st.header("Semantic search")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    from sentence_transformers import SentenceTransformer

    cols_to_use = ["title", "keywords", "description", "category"]  

    model = load_model("./modelHF")
    embeddings_db = compute_embeddings(df, cols_to_use, model, batch_size=64)
    #embeddings_db = model.encode(df["title", "keywords","description", "category", "year"].tolist()) 

    query = st.text_input("Describe what you are looking for")

    if query:
        #results = semantic_search(df, query, model, embeddings_db)
        #results = semantic_search_with_year(df, query, model, embeddings_db)
        results = semantic_search_with_year_range(df, query, model, embeddings_db, top_k=10, fallback_to_full=True)
        display_results_semantic(results)