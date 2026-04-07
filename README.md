# Document Analysis & Search Tool
RETEX Project — POC1 — Phase 1

This project is a Streamlit-based application designed to analyze documents, extract metadata using an LLM, classify them with a machine learning model, and perform multiple types of search:
- Lexical Search
- Combined Search (category + year + keywords)
- Semantic Search (Sentence Transformers)

It supports PDF, DOCX, PPTX, XLSX, and TXT files.

## I. Features

### 1. Document Analysis
Extracts raw text from uploaded files
Uses an LLM to generate:
-> Title
-> Description
-> Keywords
-> Category
-> Year

Cross-checks the LLM category with an ML classifier
Automatically stores the final result in the database

### 2. Document Database
A simple CSV-based database storing:
**Column | *Description***
title	| *Document title*
description	| *Summary extracted by the LLM*
keywords |	*Comma-separated keywords*
category	| *Final category (LLM + ML arbitration)*
year | *Year extracted from the document*


### 3. Lexical Search
Keyword-based search across all fields.

### 4. Combined Search
Filter by:
- Category
- Year range
- Keywords

### 5. Semantic Search
Uses a SentenceTransformer model to compute embeddings and retrieve the most semantically similar documents.

## II. Project Structure

/demo1-V2
├── core/
|    ├── text_extractor.py
|    ├── llm_analyzer.py
|    ├── ml_classifier.py
|    ├── database.py
|    ├── embedding_helper.py
|    ├── search.py
|    └── config.py
├── models/
|    ├── modele_ml_lr.joblib
|    ├── movies_database.csv
|    ├── train_ml_model.py
|    └── vectorizer2.joblib
├── .gitignore
├── README.md
├── main_en.py
├── main_en_search.py
├── requirements.txt
└── test_HarryPotter1.docx

## III. Running the App
bash >> streamlit run app.py

## IV. Requirements
Python 3.9+
Streamlit
pandas
scikit-learn
sentence-transformers
pdfplumber
python-docx
python-pptx

📄 License
MIT License.
