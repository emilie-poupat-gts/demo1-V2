from core.search import semantic_search

def build_rag_documents(df):
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}\n"
            f"Year: {row['year']}\n"
            f"Category: {row['category']}\n"
            f"Keywords: {row['keywords']}\n"
            f"Description: {row['description']}"
        )
        docs.append(text)
    return docs

def rag_retrieve(query, model, embeddings_db, df, top_k=5):
    # Recherche sémantique classique
    results = semantic_search(df, query, model, embeddings_db, top_k=top_k)

    # Convertir les résultats en documents textuels
    docs = []
    for _, row in results.iterrows():
        doc = (
            f"Title: {row['title']}\n"
            f"Year: {row['year']}\n"
            f"Category: {row['category']}\n"
            f"Keywords: {row['keywords']}\n"
            f"Description: {row['description']}"
        )
        docs.append(doc)

    return docs

def build_rag_prompt(question, retrieved_docs):
    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"""
Tu es un assistant expert en analyse de bases de films.

Voici des extraits du dataset :

{context}

Question utilisateur :
{question}

Analyse la question en t'appuyant uniquement sur les documents ci-dessus.
Donne une réponse synthétique, structurée et précise.
"""
    return prompt
