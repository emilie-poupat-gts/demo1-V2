📚 Document Analysis & Search Tool
RETEX Project — POC1 — Phase 1
This project is a Streamlit-based application designed to analyze documents, extract metadata using an LLM, classify them with a machine learning model, and perform multiple types of search:

Lexical Search

Combined Search (category + year + keywords)

Semantic Search (Sentence Transformers)

It supports PDF, DOCX, PPTX, XLSX, and TXT files.

🚀 Features
🔍 Document Analysis
Extracts raw text from uploaded files

Uses an LLM to generate:

Title

Description

Keywords

Category

Year

Cross-checks the LLM category with an ML classifier

Automatically stores the final result in the database

🗂️ Document Database
A simple CSV-based database storing:

Column	Description
title	Document title
description	Summary extracted by the LLM
keywords	Comma-separated keywords
category	Final category (LLM + ML arbitration)
year	Year extracted from the document


🔍 Lexical Search
Keyword-based search across all fields.

🧩 Combined Search
Filter by:

Category

Year range

Keywords

🧠 Semantic Search
Uses a SentenceTransformer model to compute embeddings and retrieve the most semantically similar documents.

🏗️ Project Structure
Code
core/
│
├── text_extractor.py
├── llm_analyzer.py
├── ml_classifier.py
├── database.py
├── search.py
└── config.py

app.py  (Streamlit main app)
database.csv
modelHF/  (SentenceTransformer model)
▶️ Running the App
bash
streamlit run app.py
📦 Requirements
Python 3.9+

Streamlit

pandas

scikit-learn

sentence-transformers

pdfplumber

python-docx

python-pptx

🤝 Contributions
Contributions, improvements, and suggestions are welcome.

📄 License
MIT License.