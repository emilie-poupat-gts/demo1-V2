import joblib
from langchain_openai import ChatOpenAI

# Modèle LLM local (Ollama API compatible OpenAI)
OPENAI_BASE_URL = "http://localhost:11434/v1"
OPENAI_API_KEY = "ollama"
GENERATOR_MODEL_NAME = "llama3.1:70b"

llm = ChatOpenAI(
    model=GENERATOR_MODEL_NAME,
    temperature=0,
    max_tokens=None,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# Modèles ML
clf = joblib.load("models/modele_ml_lr.joblib")
vectorizer = joblib.load("models/vectorizer2.joblib")

# Catégories autorisées
CATEGORIES = [
    "Sci-Fi",
    "Action",
    "Thriller",
    "Fantasy",
    "Animation",
    "Comédie",
    "Horreur"
]

CSV_PATH = "models/movies_database.csv"
