import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://www.googleapis.com/customsearch/v1",
    params={
        "key": os.getenv("GOOGLE_API_KEY"),
        "cx":  os.getenv("GOOGLE_CSE_ID"),
        "q":   "Fed interest rates",
        "num": 3,
    }
)

print("Statut :", response.status_code)
print("Réponse :", response.json())