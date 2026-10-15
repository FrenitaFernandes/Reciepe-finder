# RecipeFinder AI

RecipeFinder AI is a Flask web application for finding recipes with semantic
FAISS search and optional Groq-powered cooking suggestions. It searches the
existing recipe metadata without changing the source CSV files or vector index.

## Features

- Search by dish, recipe topic, or ingredient.
- Ingredient matching with matched and still-needed ingredient details.
- Recipe ingredients and cooking instructions from the dataset.
- FAISS and Sentence Transformers semantic search.
- Optional Groq AI recommendations based only on returned database recipes.
- Markdown-aware, safely escaped AI rendering for headings, lists, bold text,
  and tables.

## Technologies

Python, Flask, Flask-CORS, FAISS, Sentence Transformers, pandas, Groq,
HTML, CSS, and JavaScript.

## Project structure

```text
backend/app.py             Flask API
backend/prepare_data.py    Optional index-generation script
frontend/index.html        Page structure
frontend/style.css         Website styling
frontend/script.js         Search and rendering logic
data/cleaned_recipes.csv   Cleaned recipe source data
data/recipes.csv           Original recipe data
data/recipes.index         FAISS index
data/recipes_metadata.pkl  Recipe metadata used by the API
```

## Windows installation and configuration

Open PowerShell in the project directory and activate the existing
environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the project Python directly instead:

```powershell
.\venv\Scripts\python.exe -m pip --version
```

The project currently has its runtime packages in the existing virtual
environment. Do not put API keys in source files.

Create or update `.env` with:

```text
GROQ_API_KEY=your_groq_api_key
```

The key is optional for recipe search. Without it, `/api/ai-suggest` reports
that AI suggestions are unavailable while search remains usable.

## Start the application

Start the backend from the project directory:

```powershell
.\venv\Scripts\python.exe backend\app.py
```

Then open `frontend\index.html` in a browser. The frontend calls the backend
at `http://127.0.0.1:5000`.

## Test the APIs

With the backend running, use a second PowerShell window:

```powershell
curl.exe http://127.0.0.1:5000/api/health
curl.exe "http://127.0.0.1:5000/api/search?q=potatoes%2C%20onion%2C%20tomato"
curl.exe "http://127.0.0.1:5000/api/search?q=chicken%20biryani"
curl.exe http://127.0.0.1:5000/api/recipes/0
```

`/api/ai-suggest` is a POST endpoint and requires JSON containing a query or
ingredients plus recipe candidates returned by `/api/search`.

## Known limitations

- Search quality depends on the existing FAISS index and metadata quality.
- Some dataset rows have incomplete ingredients or instructions.
- Groq suggestions require network access, a valid key, and model access.
- The frontend is a static page and currently expects the backend on port 5000.
- AI output is safely rendered for common Markdown structures; arbitrary
  embedded HTML is treated as text rather than executable markup.
