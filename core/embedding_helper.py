def build_corpus(df, cols):
    corpus=[]
    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}. "
            f"Description: {row['description']}. "
            f"Keywords: {row['keywords']}. "
            f"Category: {row['category']}. "
            f"Year: {row['year']}."            
        )
        corpus.append(text)
    return corpus

def load_model(path="./modelHF"):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(path)

def compute_embeddings(df, cols, model, batch_size=64):
    corpus = build_corpus(df, cols)
    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings