import pandas as pd
from pathlib import Path

data_dir = Path("data")

# Load both cleaned datasets
original = pd.read_csv(data_dir / "cleaned_recipes.csv")
indian = pd.read_csv(data_dir / "indian_recipes_cleaned.csv")

# Standardize original recipes
original_data = pd.DataFrame({
    "recipe_name": original["Title"],
    "ingredients": original["Cleaned_Ingredients"],
    "instructions": original["Instructions"],
    "prep_time": pd.NA,
    "cook_time": pd.NA,
    "total_time": pd.NA,
    "servings": pd.NA,
    "cuisine": pd.NA,
    "course": pd.NA,
    "diet": pd.NA,
    "recipe_name_original": original["Title"],
    "ingredients_original": original["Cleaned_Ingredients"],
    "instructions_original": original["Instructions"],
    "source_dataset": "Original",
    "source_url": pd.NA
})

# Standardize Indian recipes
indian_data = pd.DataFrame({
    "recipe_name": indian["TranslatedRecipeName"],
    "ingredients": indian["TranslatedIngredients"],
    "instructions": indian["TranslatedInstructions"],
    "prep_time": indian["PrepTimeInMins"],
    "cook_time": indian["CookTimeInMins"],
    "total_time": indian["TotalTimeInMins"],
    "servings": indian["Servings"],
    "cuisine": indian["Cuisine"],
    "course": indian["Course"],
    "diet": indian["Diet"],
    "recipe_name_original": indian["RecipeName"],
    "ingredients_original": indian["Ingredients"],
    "instructions_original": indian["Instructions"],
    "source_dataset": "Indian",
    "source_url": indian["URL"]
})

# Combine datasets
combined = pd.concat([original_data, indian_data], ignore_index=True)

# Generate unique IDs
combined.insert(
    0, "recipe_id",
    [f"REC{i:06d}" for i in range(1, len(combined) + 1)]
)

# Save new dataset
combined.to_csv(
    data_dir / "combined_recipes.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCOMBINATION COMPLETED")
print("Original recipes:", len(original_data))
print("Indian recipes:", len(indian_data))
print("Combined recipes:", len(combined))
print("Columns:", len(combined.columns))
print("Missing names:", combined["recipe_name"].isna().sum())
print("Missing ingredients:", combined["ingredients"].isna().sum())
print("Missing instructions:", combined["instructions"].isna().sum())
print("\nSaved: data/combined_recipes.csv")
