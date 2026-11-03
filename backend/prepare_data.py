import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

data_path = DATA_DIR / "combined_recipes.csv"
index_path = DATA_DIR / "recipes.index"
metadata_path = DATA_DIR / "recipes_metadata.pkl"

print("Loading combined recipes dataset...")

df = pd.read_csv(data_path)

# Check required columns
required_columns = [
    "recipe_id",
    "recipe_name",
    "ingredients",
    "instructions"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Prepare searchable documents
documents = (
    "Recipe: " + df["recipe_name"].fillna("").astype(str)
    + "\nIngredients: "
    + df["ingredients"].fillna("").astype(str)
).tolist()

print("Total recipes:", len(documents))
print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating recipe embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype("float32")

print("Building FAISS index...")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, str(index_path))

# Save complete recipe metadata
with open(metadata_path, "wb") as file:
    pickle.dump(df.to_dict(orient="records"), file)

print("\nFAISS index created successfully!")
print("Total recipes indexed:", index.ntotal)
print("Embedding dimensions:", embeddings.shape[1])
print("Original dataset recipes:", len(df[df["source_dataset"] == "Original"]))
print("Indian dataset recipes:", len(df[df["source_dataset"] == "Indian"]))
print("Index saved to:", index_path)
print("Metadata saved to:", metadata_path)