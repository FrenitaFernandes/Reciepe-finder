
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import os
import ast
import re
import faiss
import pickle

# --------------------------------------------------
# 1. Initialize Flask
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# 2. Project paths and environment variables
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "recipes.index"
METADATA_PATH = DATA_DIR / "recipes_metadata.pkl"

load_dotenv(BASE_DIR / ".env", override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

# Allow more time for Groq requests and retry once.
client = (
    Groq(
        api_key=groq_api_key,
        timeout=90.0,
        max_retries=1
    )
    if groq_api_key
    else None
)

if client:
    print("Groq client initialized successfully.")
else:
    print("Warning: GROQ_API_KEY not found. AI suggestions are disabled.")

# --------------------------------------------------
# 3. Load FAISS index and recipe data
# --------------------------------------------------

print("Loading FAISS recipe index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Recipe metadata not found: {METADATA_PATH}")

index = faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH, "rb") as file:
    recipes = pickle.load(file)

# --------------------------------------------------
# 4. Load Sentence Transformer
# --------------------------------------------------

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"RecipeFinder AI ready! {len(recipes)} recipes loaded.")

# --------------------------------------------------
# 5. Text and formatting helpers
# --------------------------------------------------

def normalize_text(text):
    text = str(text).lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ingredient(text):
    normalized = normalize_text(text)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", " ", normalized)
    normalized = re.sub(
        r"\b(?:small|medium|large|fresh|dried|chopped|sliced|diced|"
        r"finely|roughly|ground|boneless|skinless|divided|plus more)\b",
        " ",
        normalized
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def ingredient_tokens(value):
    return {
        token for token in normalize_ingredient(value).split()
        if len(token) > 1
    }


def ingredient_variants(token):
    variants = {token}
    if token.endswith("ies") and len(token) > 3:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 3:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 2:
        variants.add(token[:-1])
    return variants


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, float) and str(value) == "nan":
        return ""

    if isinstance(value, (list, tuple)):
        return "; ".join(
            clean_text(item) for item in value if clean_text(item)
        )

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return "; ".join(
                    clean_text(item) for item in parsed if clean_text(item)
                )
        except (ValueError, SyntaxError):
            pass

    text = text.replace("|", "; ")
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip(" ;")


def get_ingredient_matches(user_ingredients, recipe_ingredients):
    recipe_tokens = ingredient_tokens(recipe_ingredients)
    matched = []
    missing = []

    for ingredient in user_ingredients:
        ingredient = str(ingredient).strip()
        tokens = ingredient_tokens(ingredient)

        if not tokens:
            continue

        recipe_variants = {
            variant
            for token in recipe_tokens
            for variant in ingredient_variants(token)
        }
        requested_variants = {
            variant
            for token in tokens
            for variant in ingredient_variants(token)
        }

        if recipe_variants.intersection(requested_variants):
            matched.append(ingredient)
        else:
            missing.append(ingredient)

    return matched, missing


def parse_ingredient_list(value):
    if isinstance(value, str):
        text = value.strip()
        items = []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    items = parsed
            except (ValueError, SyntaxError):
                pass
        if not items:
            items = text.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []

    return [
        str(item).strip()
        for item in items
        if str(item).strip()
    ]


def infer_user_ingredients(query):
    if "," in query:
        return parse_ingredient_list(query)
    if len(query.split()) == 1:
        return parse_ingredient_list(query)
    return []


def get_recipe_fields(recipe):
    return {
        "title": clean_text(recipe.get("Title", "")),
        "ingredients": clean_text(recipe.get("Cleaned_Ingredients", "")),
        "instructions": clean_text(recipe.get("Instructions", ""))
    }


# --------------------------------------------------
# 6. Home route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to RecipeFinder AI!",
        "status": "running",
        "recipes_loaded": len(recipes),
        "features": [
            "FAISS semantic recipe search",
            "Ingredient matching",
            "Recipe instructions",
            "Groq AI recipe suggestions"
        ],
        "groq_enabled": client is not None
    })


# --------------------------------------------------
# 7. Health check
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "recipes_loaded": len(recipes),
        "faiss_index_size": index.ntotal,
        "groq_enabled": client is not None
    })


# --------------------------------------------------
# 8. Recipe search
# --------------------------------------------------

@app.route("/api/search", methods=["GET"])
def search_recipes():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "error": "Please enter a recipe name or search query."
        }), 400

    try:
        limit = int(request.args.get("limit", 9))
    except (ValueError, TypeError):
        limit = 9

    limit = max(1, min(limit, 20))

    raw_ingredients = request.args.get("ingredients", "").strip()
    user_ingredients = parse_ingredient_list(raw_ingredients)
    if not user_ingredients:
        user_ingredients = infer_user_ingredients(query)

    try:
        query_embedding = model.encode(
            ["Recipe: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        search_limit = min(index.ntotal, len(recipes))

        if search_limit == 0:
            return jsonify({
                "query": query,
                "count": 0,
                "results": []
            })

        scores, indices = index.search(query_embedding, search_limit)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(recipes):
                continue

            fields = get_recipe_fields(recipes[idx])
            ingredients = fields["ingredients"]

            if user_ingredients:
                matched, missing = get_ingredient_matches(
                    user_ingredients, ingredients
                )
            else:
                matched, missing = [], []

            match_count = len(matched)
            total_requested = len(user_ingredients)

            match_percentage = (
                round(match_count / total_requested * 100, 1)
                if total_requested else None
            )

            results.append({
                "id": int(idx),
                "title": fields["title"],
                "ingredients": ingredients,
                "instructions": fields["instructions"],
                "similarity": round(float(score), 4),
                "matched_ingredients": matched,
                "missing_ingredients": missing,
                "ingredient_match_count": match_count,
                "ingredient_match_percentage": match_percentage
            })

        if user_ingredients:
            results.sort(
                key=lambda item: (
                    item["ingredient_match_count"],
                    item["similarity"]
                ),
                reverse=True
            )

        selected_results = results[:limit]

        return jsonify({
            "query": query,
            "user_ingredients": user_ingredients,
            "ingredient_matching_enabled": bool(user_ingredients),
            "count": len(selected_results),
            "results": selected_results
        })

    except Exception:
        app.logger.exception("Recipe search failed")
        return jsonify({
            "error": "Recipe search failed. Check the backend terminal."
        }), 500


# --------------------------------------------------
# 9. Get a single recipe
# --------------------------------------------------

@app.route("/api/recipes/<int:recipe_id>", methods=["GET"])
def get_recipe(recipe_id):
    if recipe_id < 0 or recipe_id >= len(recipes):
        return jsonify({"error": "Recipe not found."}), 404

    fields = get_recipe_fields(recipes[recipe_id])

    return jsonify({
        "id": recipe_id,
        "title": fields["title"],
        "ingredients": fields["ingredients"],
        "instructions": fields["instructions"]
    })


# --------------------------------------------------
# 10. Groq AI suggestions
# --------------------------------------------------

@app.route("/api/ai-suggest", methods=["POST"])
def ai_suggest():
    data = request.get_json(silent=True) or {}

    user_ingredients = parse_ingredient_list(data.get("ingredients", []))
    search_query = clean_text(data.get("query", ""))[:200]
    candidate_recipes = data.get("recipes", [])

    if not user_ingredients and not search_query:
        return jsonify({
            "error": "Provide a search query or ingredients."
        }), 400

    if not isinstance(candidate_recipes, list) or not candidate_recipes:
        return jsonify({
            "error": "Provide recipes returned by the recipe search."
        }), 400

    safe_recipes = []

    for recipe in candidate_recipes[:3]:
        if not isinstance(recipe, dict):
            continue

        try:
            recipe_id = int(recipe.get("id"))
        except (TypeError, ValueError):
            continue

        if recipe_id < 0 or recipe_id >= len(recipes):
            continue

        fields = get_recipe_fields(recipes[recipe_id])
        safe_recipes.append({
            "id": recipe_id,
            "title": fields["title"][:200],
            "ingredients": fields["ingredients"][:1800],
            "instructions": fields["instructions"][:1800]
        })

    if not safe_recipes:
        return jsonify({"error": "No valid recipes were provided."}), 400

    if client is None:
        return jsonify({
            "error": "GROQ_API_KEY is missing from your .env file."
        }), 503

    prompt = f"""
You are RecipeFinder AI, a helpful cooking assistant.

Search query:
{search_query or "Not provided"}

Ingredients provided by the user:
{user_ingredients or "Not provided"}

Recipes from the existing database:
{safe_recipes}

Choose the most suitable recipe from these recipes.
Use only the supplied database details.
Use clear headings:
1. Recommended recipe
2. Why it matches
3. Ingredients needed
4. Cooking steps

Identify available and missing ingredients only from the supplied data.
Do not treat the search phrase as an ingredient unless it is explicitly listed above.
Do not invent ingredients, cooking times, or temperatures.
If information is unavailable, say so.
Use short sentences and simple beginner-friendly language.
Do not use Markdown tables, bold markers (**), italic markers (*), horizontal rules (---), or long dashes.
Use plain numbered headings and short bullet lists instead.
Remind users to check allergies and cook food safely.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are RecipeFinder AI. "
                        "Follow the supplied recipe data and never invent details."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )

        suggestion = response.choices[0].message.content

        return jsonify({
            "success": True,
            "suggestion": suggestion or "No suggestion was generated."
        })

    except Exception as error:
        app.logger.exception("Groq AI request failed")

        return jsonify({
            "error": (
                "Groq could not generate a suggestion. "
                "Check the network, API key, model access, and API limits. "
                "See the backend terminal for the specific error."
            )
        }), 502


# --------------------------------------------------
# 11. Start Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
