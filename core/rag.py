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
    # Classic semantic seard
    results = semantic_search(df, query, model, embeddings_db, top_k=top_k)

    # Convert results in text documents
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
You are an assistant specialized in cinema.

Here are extracts from the dataset :

{context}

User question :
{question}

Analyze the question based only on the documents above. 
Give a synthetic, structured and precise response.

"""
    return prompt
