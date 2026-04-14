import streamlit as st
import pandas as pd

from core.text_extractor import extract_text
from core.llm_analyzer import analyze_with_llm
from core.ml_classifier import classify
from core.database import load_database, add_entry
from core.search import lexical_search, combined_search, semantic_search
from core.config import llm, clf, vectorizer

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Document tool", layout="wide")
st.title("Document tool — contribution & search")
st.markdown("RETEX Project — POC1 — Phase 1")


# ---------------------------------------------------------
# CACHE MODELS & EMBEDDINGS
# ---------------------------------------------------------
@st.cache_resource
def load_st_model():
    return SentenceTransformer("./modelHF")

@st.cache_data
def compute_embeddings(descriptions):
    model = load_st_model()
    return model.encode(descriptions)


# ---------------------------------------------------------
# LOAD DATABASE
# ---------------------------------------------------------
df = load_database()

st.sidebar.header("Main menu")
mode = st.sidebar.radio(
    "Choose a mode:",
    [
        " Show database",
        " Add a document"
    ]
)


# ---------------------------------------------------------
# RESULT DISPLAY
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
# MODE 1 — ADD DOCUMENT
# ---------------------------------------------------------
if mode == " Add a document":
    st.header("Analysis and addition of documents in the database")

    if "df" not in st.session_state:
        st.session_state.df = load_database()

    with st.expander("View current database"):
        st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'pptx', 'xlsx', 'txt']
    )

    if uploaded_file is not None:
        if st.button("Analyze and add to database"):

            # Text extraction
            with st.spinner("Extracting text..."):
                try:
                    extracted_text = extract_text(uploaded_file)
                except Exception as e:
                    st.error(f"Extraction error: {e}")
                    extracted_text = None

            if extracted_text:

                # LLM analysis (summarization + extraction)
                with st.spinner("Analyzing with LLM..."):
                    try:
                        llm_data = analyze_with_llm(extracted_text, llm)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
                        llm_data = None

                if llm_data:
                    st.subheader("LLM Extracted Data:")
                    st.json(llm_data)

                    # ML arbitration
                    with st.spinner("Comparing with ML model..."):
                        desc_vec = vectorizer.transform([llm_data['description']])
                        probas = clf.predict_proba(desc_vec)[0]
                        max_proba = probas.max()
                        ml_pred_category = clf.classes_[probas.argmax()]

                        llm_category = llm_data['category']
                        final_category = llm_category

                        if llm_category == ml_pred_category:
                            explanation = (
                                f"LLM and ML agree: **{llm_category}** "
                                f"(ML confidence: {max_proba:.2f})"
                            )
                        else:
                            explanation = (
                                f"Conflict: LLM = '{llm_category}', ML = '{ml_pred_category}' "
                                f"(ML confidence: {max_proba:.2f})\n\n"
                            )

                            if max_proba >= 0.7:
                                final_category = ml_pred_category
                                explanation += (
                                    f"ML is very confident ({max_proba:.2f}). "
                                    f"Final category: **{ml_pred_category}**"
                                )
                            else:
                                explanation += (
                                    f"ML is not confident ({max_proba:.2f}). "
                                    f"Keeping LLM category: **{llm_category}**"
                                )

                        st.info(explanation)

                    # Add to database
                    new_entry = {
                        "title": llm_data["title"],
                        "description": llm_data["description"],
                        "keywords": ", ".join(llm_data["keywords"]),
                        "category": final_category,
                        "year": llm_data["year"]
                    }

                    st.session_state.df = add_entry(st.session_state.df, new_entry)
                    st.success(f"The document **{llm_data['title']}** has been added to the database!")

            else:
                st.error("Unable to extract text from this document.")


# ---------------------------------------------------------
# MODE 2 — SHOW DATABASE
# ---------------------------------------------------------
elif mode == " Show database":
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

    model = load_st_model()
    embeddings_db = compute_embeddings(df["description"].tolist())

    query = st.text_input("Describe what you are looking for")

    if query:
        results = semantic_search(df, query, model, embeddings_db)
        display_results(results)
