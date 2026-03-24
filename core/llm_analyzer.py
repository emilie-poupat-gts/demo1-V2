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
Tu es un assistant spécialisé en cinéma. Analyse le texte suivant et extrais les métadonnées d'un film.
Tu DOIS répondre UNIQUEMENT au format JSON valide. Aucun texte avant ou après.
La catégorie doit être dans : {CATEGORIES}.
Si l'année est introuvable, mets "0000".

Format attendu :
{{
    "title": "Titre du film",
    "description": "Résumé clair en une phrase",
    "keywords": ["mot1", "mot2"],
    "category": "Catégorie",
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
