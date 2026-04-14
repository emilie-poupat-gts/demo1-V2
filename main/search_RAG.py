import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time


from sentence_transformers import SentenceTransformer

from core.text_extractor import extract_text
from core.llm_analyzer import analyze_with_llm, rag_answer_with_llm
from core.ml_classifier import classify
from core.database import load_database, add_entry
from core.search import (
    lexical_search,
    combined_search,
    semantic_search_with_year_range
)
from core.embedding_helper import (
    compute_embeddings  
)
from core.config import llm, clf, vectorizer
from core.rag import (
    build_rag_prompt  
)

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
        " Semantic search",
        "RAG"
    ]
)

# ---------------------------------------------------------
# CACHED MODEL, EMBEDDINGS AND FAISS INDEX
# ---------------------------------------------------------
@st.cache_resource
def get_embedding_model(path: str = "./modelHF"):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(path)

@st.cache_resource
def get_db_embeddings(df: pd.DataFrame):
    """Compute and cache embeddings for the whole database."""
    model = get_embedding_model("./modelHF")
    cols_to_use = ["title", "keywords", "description", "category"]
    embeddings = compute_embeddings(df, cols_to_use, model, batch_size=64)
    return embeddings

@st.cache_resource
def get_faiss_index(embeddings: np.ndarray):
    """Build and cache a FAISS index over the embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index

# ---------------------------------------------------------
# RESULT DISPLAY FUNCTIONS
# ---------------------------------------------------------
def display_results_semantic(results: pd.DataFrame):
    if results.empty:
        st.warning("No results found")
        return

    st.success(f"{len(results)} result(s) found")

    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['year']}) — {row['category']} "):
            st.write(f"**Match score:** `{row['score']:.4f}`")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Keywords:** {row['keywords']}")

def display_results_classic(results: pd.DataFrame):
    if results.empty:
        st.warning("No results found")
        return

    st.success(f"{len(results)} result(s) found")

    for _, row in results.iterrows():
        with st.expander(f"**{row['title']}** ({row['year']}) — {row['category']} "):
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Keywords:** {row['keywords']}")

# ---------------------------------------------------------
# FAST RAG RETRIEVAL USING FAISS
# ---------------------------------------------------------
def rag_retrieve_fast(query: str, model, index, df: pd.DataFrame, top_k: int = 8):
    """Retrieve top_k documents using FAISS index."""
    q_emb = model.encode([query])
    scores, idx = index.search(q_emb, top_k)

    retrieved_rows = []
    for i, score in zip(idx[0], scores[0]):
        row = df.iloc[i]
        retrieved_rows.append((row, float(score)))
    return retrieved_rows

def build_rag_documents_from_rows(retrieved_rows):
    """Turn retrieved rows into text snippets for the RAG prompt."""
    docs = []
    for row, score in retrieved_rows:
        doc_text = (
            f"Title: {row['title']}\n"
            f"Year: {row['year']}\n"
            f"Category: {row['category']}\n"
            f"Keywords: {row['keywords']}\n"
            f"Description: {row['description']}\n"
        )
        docs.append(doc_text)
    return docs

# ---------------------------------------------------------
# MODE 1 — DATABASE
# ---------------------------------------------------------
if mode == " Show database":
    st.header("Database")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# MODE 2 — LEXICAL SEARCH
# ---------------------------------------------------------
elif mode == " Lexical search":
    st.header("Lexical search")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    query = st.text_input("Type a word or sentence", "sci-fi")

    if query:
        results = lexical_search(df, query)
        display_results_classic(results)

# ---------------------------------------------------------
# MODE 3 — COMBINED SEARCH
# ---------------------------------------------------------
elif mode == " Combined search (with filters)":
    st.header("Combined search")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    categories = sorted(df["category"].unique())
    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    col1, col2 = st.columns(2)

    with col1:
        selected_categories = st.multiselect("Categories:", categories)

    with col2:
        selected_years = st.slider("Years:", year_min, year_max, (year_min, year_max))

    keywords = st.text_input("Select keywords (separated by a comma):")
    selected_keywords = sorted(list(set(",".join(keywords).split(",")))) if keywords else []

    if st.button("Filter"):
        results = combined_search(df, selected_categories, selected_years, selected_keywords)
        display_results_classic(results)

# ---------------------------------------------------------
# MODE 4 — SEMANTIC SEARCH
# ---------------------------------------------------------
elif mode == " Semantic search":
    st.header("Semantic search")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    model = get_embedding_model("./modelHF")
    embeddings_db = get_db_embeddings(df)

    query = st.text_input("Describe what you are looking for", "sci-fi after 2020")

    if query:
        results = semantic_search_with_year_range(
            df,
            query,
            model,
            embeddings_db,
            top_k=10,
            fallback_to_full=True
        )

        st.session_state["semantic_results"] = results
    
    if "semantic_results" in st.session_state:
        results = st.session_state["semantic_results"]

    # Filters after the semantic search 
    categories = sorted(results["category"].unique())
    selected_cat = st.multiselect("Filter by category", categories)

    min_year, max_year = int(results["year"].min()), int(results["year"].max())
    selected_years = st.slider("Filter by year", min_year, max_year, (min_year, max_year))

    filtered = results.copy()

    if selected_cat:
        filtered = filtered[filtered["category"].isin(selected_cat)]

    filtered = filtered[
        (filtered["year"] >= selected_years[0]) &
        (filtered["year"] <= selected_years[1])
    ]

    st.subheader("Filtered results")
    st.dataframe(filtered, use_container_width=True)


# ---------------------------------------------------------
# MODE 5 — RAG
# ---------------------------------------------------------
elif mode == "RAG":
    st.header("RAG — Retrieval Augmented Generation")

    query = st.text_input("Ask a question about the dataset", "Are there any sci‑fi movies released before 2000?")

    if query:
        progress = st.progress(0)
        log = st.empty()

        # Load the embedding model
        t0 = time.time()
        model = get_embedding_model("./modelHF")
        progress.progress(10)
        #log.write("Model loaded")

        # Load embeddings
        embeddings_db = get_db_embeddings(df)
        progress.progress(30)
        #log.write(f"Embeddings loaded in {time.time() - t0:.2f}s")

        # Load FAISS index
        t1 = time.time()
        index = get_faiss_index(embeddings_db)
        progress.progress(50)
        #log.write(f"FAISS index ready in {time.time() - t1:.2f}s")

        # Retrieve documents
        t2 = time.time()
        retrieved_rows = rag_retrieve_fast(query, model, index, df, top_k=25)
        progress.progress(70)
        #log.write(f"Retrieval done in {time.time() - t2:.2f}s")

        # Build RAG docs
        t3 = time.time()
        rag_docs = build_rag_documents_from_rows(retrieved_rows)
        progress.progress(80)
        #log.write(f"Document formatting done in {time.time() - t3:.2f}s")

        # Call LLM 
        t4 = time.time()
        prompt = build_rag_prompt(query, rag_docs)
        answer = rag_answer_with_llm(prompt, llm)
        progress.progress(100)
        #log.write(f"LLM answered in {time.time() - t4:.2f}s")

        st.subheader("LMM response")
        st.write(answer)

        with st.expander("Documents used"):
            for doc in rag_docs:
                st.markdown(doc.replace("\n", "  \n"))

        with st.expander("Debug timing"):
            st.write({
                "Model load": f"{t1 - t0:.2f}s",
                "Embedding load": f"{t2 - t1:.2f}s",
                "FAISS index": f"{t3 - t2:.2f}s",
                "Retrieval": f"{t4 - t3:.2f}s",
                "LLM call": f"{time.time() - t4:.2f}s",
            })

