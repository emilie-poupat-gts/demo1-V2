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

### 6. RAG


## II. Project Structure

/demo1-V2
├── core/
|    ├── config.py
|    ├── database.py
|    ├── embedding_helper.py
|    ├── llm_analyzer.py
|    ├── ml_classifier.py
|    ├── rag.py
|    ├── search.py
|    └── text_extractor.py
├── main/
|    ├── add.py
|    └── search_RAG.py
├── models/
|    ├── modele_ml_lr.joblib
|    ├── train_ml_model.py
|    └── vectorizer2.joblib
├── src/
|    ├── movies_database.csv
|    └── test_HarryPotter1.docx
├── .gitignore
├── README.md
└── requirements.txt



In the .gitignore, there is the folder *ModelHF*. It's the model Hugging Face used for the semantic search. It's in a folder because the VM doesn't have access to internet so to the Hugging Face's models, we have to put it in a folder. 
==> In an environment that has access to internet, we can directly call the Hugging Face's model : *sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2*.

## III. Running the App
bash >> streamlit run main/add.py (for the add.py demo)
bash >> streamlit run main/search_RAG.py (for the search_RAG.py demo)

If in a venv :
bash (venv) >> python -m streamlit run <file>.py

## IV. Requirements
see the requirements.txt file

📄 License
MIT License.
