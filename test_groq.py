
import requests

url = "http://127.0.0.1:5000/api/ai-suggest"

payload = {
    "ingredients": ["chicken", "onion", "tomato"],
    "recipes": [
        {
            "title": "Chicken and Tomato Stew",
            "ingredients": "chicken; onion; tomato; salt; oil",
            "instructions": "1. Chop the onion and tomato. 2. Cook the chicken with oil. 3. Add onion and tomato, then simmer until cooked."
        }
    ]
}

try:
    response = requests.post(url, json=payload, timeout=45)
    print("Status code:", response.status_code)
    print("Response:", response.text)
except requests.exceptions.Timeout:
    print("ERROR: Groq request took longer than 45 seconds.")
except requests.exceptions.RequestException as error:
    print("ERROR:", error)
