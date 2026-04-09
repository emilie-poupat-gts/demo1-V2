import json
import re
from core.config import CATEGORIES

import json
import re
from core.config import CATEGORIES

def analyze_with_llm(text, llm):

    def split_text(t, size=2500):
        return [t[i:i+size] for i in range(0, len(t), size)]

    chunks = split_text(text)
    results = []

    for chunk in chunks:
        prompt = f"""
You are an assistant specialized in cinema. Analyze the following text and extract the metadata of a movie.
You MUST respond ONLY in valid JSON format. No text before or after.
The category must be one of : {CATEGORIES}.
If the year can't be found, use "OOOO".

Expected format:
{{
    "title": "Title of the movie",
    "description": "Short and pertinent description of the movie in 1 sentence",
    "keywords": ["word1", "word2"],
    "category": "Category",
    "year": 2020
}}

Text :
\"\"\"
{chunk}
\"\"\"
"""

        response = llm.invoke(prompt)
        raw = response.content

        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if match:
            try:
                results.append(json.loads(match.group(0)))
            except:
                pass

    if not results:
        raise ValueError("No valid JSON found")

    fusion = {
        "title": results[0]["title"],
        "description": " ".join(r["description"] for r in results),
        "keywords": list({m for r in results for m in r["keywords"]}),
        "category": results[0]["category"],
        "year": results[0].get("year", "0000")
    }

    return fusion

def rag_answer_with_llm(prompt, llm):
    """
    LLM call for RAG question answering.
    No JSON, no metadata extraction.
    """
    response = llm.invoke(prompt)
    return response.content

