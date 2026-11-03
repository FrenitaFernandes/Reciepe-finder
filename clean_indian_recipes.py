
import pandas as pd
import re
from pathlib import Path

input_file = Path("data/IndianFoodDatasetCSV.csv")
output_file = Path("data/indian_recipes_cleaned.csv")

if not input_file.exists():
    print("ERROR: Indian recipe dataset not found.")
    raise SystemExit(1)

df = pd.read_csv(input_file, encoding="utf-8")
print("Original rows:", len(df))

# Clean column names
df.columns = df.columns.str.strip()

# Clean whitespace in text columns
for col in df.select_dtypes(include=["str", "object"]).columns:
    df[col] = df[col].apply(
        lambda value: re.sub(r"\s+", " ", value).strip()
        if isinstance(value, str) else value
    )

# Remove rows missing essential recipe information
required_columns = [
    "RecipeName",
    "TranslatedRecipeName",
    "Ingredients",
    "TranslatedIngredients",
    "Instructions",
    "TranslatedInstructions",
]

for col in required_columns:
    df[col] = df[col].replace("", pd.NA)

before = len(df)
df = df.dropna(subset=required_columns)
print("Rows removed for missing essential information:", before - len(df))

# Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", before - len(df))

# Remove repeated instruction text when an earlier complete
# sentence sequence appears again later in the same instructions.
def clean_repeated_instructions(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    # Split at sentence boundaries and clean each sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        key = re.sub(r"\W+", " ", sentence.lower()).strip()

        if key and key not in seen:
            kept.append(sentence)
            seen.add(key)

    return " ".join(kept)

for col in ["Instructions", "TranslatedInstructions"]:
    df[col] = df[col].apply(clean_repeated_instructions)

# Remove duplicate translated names, keeping the first occurrence
before = len(df)
df = df.drop_duplicates(subset=["TranslatedRecipeName"], keep="first")
print("Duplicate translated recipe names removed:", before - len(df))

# Save to a new cleaned file; original data remains unchanged.
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Final cleaned rows:", len(df))
print("Saved to:", output_file)
print("Original dataset remains unchanged.")
