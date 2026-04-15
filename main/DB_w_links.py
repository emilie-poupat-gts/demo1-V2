import streamlit as st
import pandas as pd
import os

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
df = pd.read_csv("src/test_links/DB_links.csv", sep=",")

st.sidebar.header("Main menu")
mode = st.sidebar.radio(
    "Choose a mode:",
    [
        " Show database",
        " Show database with link"
    ]
)

# ---------------------------------------------------------
# GLOBAL CSS FOR COMPACT ROWS + HOVER + SMALL BUTTONS
# ---------------------------------------------------------
st.markdown("""
<style>

.row-tight {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
}

.row-tight:hover {
    background-color: #e6f2ff !important;
}

.stButton>button, .stDownloadButton>button {
    padding: 2px 6px !important;
    font-size: 0.75rem !important;
    line-height: 0.9rem !important;
}

div[data-testid="column"] {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}

.desc-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

hr {
    margin: 2px 0 !important;
}

/* Sticky header */
.sticky-header {
    position: sticky;
    top: 0;
    background-color: white;
    padding: 6px 0;
    z-index: 999;
    border-bottom: 2px solid #ddd;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODE 1 — DATABASE NOT CLICKABLE
# ---------------------------------------------------------
if mode == " Show database":
    st.header("Database")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# MODE 2 — DATABASE WITH LINK + SEARCH + SORTING
# ---------------------------------------------------------
if mode == " Show database with link":
    st.header("Database with link clickable")

    # -------------------------------
    # SEARCH BAR
    # -------------------------------
    search_query = st.text_input("Lexical search in database", "")

    if search_query:
        df = df[
            df.apply(lambda row:
                search_query.lower() in str(row["title"]).lower() or
                search_query.lower() in str(row["description"]).lower() or
                search_query.lower() in str(row["category"]).lower() or
                search_query.lower() in str(row["keywords"]).lower() or
                search_query.lower() in str(row["year"]).lower(),
                axis=1
            )
        ]

    # -------------------------------
    # SORTING
    # -------------------------------
    sort_col = st.selectbox("Sort by:", ["title", "year", "category"])
    sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)

    df = df.sort_values(
        by=sort_col,
        ascending=(sort_order == "Ascending")
    )

    # -------------------------------
    # STICKY HEADER
    # -------------------------------
    st.markdown("""
    <div class="sticky-header">
        <div style="display:flex; font-weight:bold;">
            <div style="flex:2;">Title</div>
            <div style="flex:4;">Description</div>
            <div style="flex:2;">Category</div>
            <div style="flex:1;">Year</div>
            <div style="flex:2;">Document</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # ROWS
    # -------------------------------
    for idx, row in df.iterrows():

        bg_color = "#f7f7f7" if idx % 2 == 0 else "#ffffff"

        full_desc = row["description"]
        short_desc = full_desc[:120] + "..." if len(full_desc) > 120 else full_desc
        #toggle_key = f"toggle_{idx}"
        unique_id = row["file_path"] #stable unique key

        with st.container():
            st.markdown(
                f"<div class='row-tight' style='background-color:{bg_color}; padding:4px;'>",
                unsafe_allow_html=True
            )

            col1, col2, col3, col4, col5 = st.columns([2, 4, 2, 1, 2])

            # Title
            col1.write(row["title"])

            # Description + show more/less
            if len(full_desc) > 120:
                if st.session_state.get(unique_id, False):
                    col2.write(full_desc)
                    if col2.button("Show less", key=f"less_{unique_id}"):
                        st.session_state[unique_id] = False
                else:
                    col2.markdown(f"<div class='desc-text'>{short_desc}</div>", unsafe_allow_html=True)
                    if col2.button("Show more", key=f"more_{unique_id}"):
                        st.session_state[unique_id] = True
            else:
                col2.markdown(f"<div class='desc-text'>{full_desc}</div>", unsafe_allow_html=True)
                
            # Category
            col3.write(row["category"])

            # Year
            col4.write(row["year"])

            # Download button
            file_path = row["file_path"]
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    col5.download_button(
                        label="Download",
                        data=f,
                        file_name=os.path.basename(file_path),
                        key=f"open_{idx}"
                    )
            else:
                col5.error("Not found")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
