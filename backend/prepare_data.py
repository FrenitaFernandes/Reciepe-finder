
import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project paths
data_path = Path("data/cleaned_recipes.csv")
index_path = Path("data/recipes.index")
metadata_path = Path("data/recipes_metadata.pkl")

print("Loading cleaned recipes...")

df = pd.read_csv(data_path)

# Combine recipe title and ingredients for searching
documents = (
    "Recipe: " + df["Title"].fillna("").astype(str)
    + "\nIngredients: "
    + df["Cleaned_Ingredients"].fillna("").astype(str)
).tolist()

print("Total recipes:", len(documents))
print("Loading embedding model...")

# Downloads the model the first time if needed
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating recipe embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Build the FAISS similarity-search index
embeddings = embeddings.astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save the index and recipe information
faiss.write_index(index, str(index_path))

with open(metadata_path, "wb") as file:
    pickle.dump(df.to_dict(orient="records"), file)

print("\nFAISS index created successfully!")
print("Indexed recipes:", index.ntotal)
print("Index saved to:", index_path)
print("Metadata saved to:", metadata_path)
