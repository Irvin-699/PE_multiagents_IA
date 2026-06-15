"""
test_api_keys.py
Teste toutes les clés API du projet P26.
Lance depuis la racine du projet : python test_api_keys.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def test_finnhub():
    print("\n--- TEST FINNHUB ---")
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        print("❌ FINNHUB_API_KEY absente du .env")
        return
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": key},
            timeout=10,
        )
        if response.status_code == 200:
            articles = response.json()
            print(f"✅ Finnhub OK — {len(articles)} articles reçus")
            if articles:
                print(f"   Exemple : {articles[0].get('headline', '')[:80]}")
        elif response.status_code == 401:
            print("❌ Clé Finnhub invalide")
        else:
            print(f"❌ Erreur Finnhub : statut {response.status_code}")
    except Exception as e:
        print(f"❌ Finnhub injoignable : {e}")


def test_newsapi():
    print("\n--- TEST NEWSAPI ---")
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        print("❌ NEWSAPI_KEY absente du .env")
        print("   → Va sur newsapi.org → Get API Key → copie la clé dans ton .env")
        return
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q"        : "Fed interest rates",
                "language" : "en",
                "pageSize" : 3,
                "apiKey"   : key,
            },
            timeout=10,
        )
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            print(f"✅ NewsAPI OK — {len(articles)} articles reçus")
            for a in articles:
                print(f"   - {a['title'][:80]}")
        elif response.status_code == 401:
            print("❌ Clé NewsAPI invalide — vérifie NEWSAPI_KEY dans le .env")
        elif response.status_code == 426:
            print("❌ Plan gratuit dépassé — attends demain (100 req/jour)")
        else:
            print(f"❌ Erreur NewsAPI : {response.status_code} — {response.json()}")
    except Exception as e:
        print(f"❌ NewsAPI injoignable : {e}")


def test_anthropic():
    print("\n--- TEST ANTHROPIC (Claude Haiku) ---")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("❌ ANTHROPIC_API_KEY absente du .env")
        return
    try:
        from anthropic import Anthropic
        client   = Anthropic(api_key=key)
        response = client.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 30,
            messages   = [{"role": "user", "content": "Réponds juste : API OK"}],
        )
        print(f"✅ Anthropic OK — Réponse : {response.content[0].text.strip()}")
    except ImportError:
        print("❌ Package 'anthropic' non installé — lance : pip install anthropic")
    except Exception as e:
        if "401" in str(e) or "authentication" in str(e).lower():
            print("❌ Clé Anthropic invalide ou solde insuffisant")
            print("   → Vérifie sur console.anthropic.com → Billing")
        else:
            print(f"❌ Erreur Anthropic : {e}")


def test_chromadb():
    print("\n--- TEST CHROMADB ---")
    try:
        import chromadb
        client      = chromadb.PersistentClient(path="./data/models/chroma")
        collections = client.list_collections()
        print(f"✅ ChromaDB OK — {len(collections)} collection(s) existante(s)")
        if collections:
            for col in collections:
                print(f"   - {col.name}")
        else:
            print("   (base vide — normal au premier lancement)")
    except ImportError:
        print("❌ ChromaDB non installé — lance : pip install chromadb")
    except Exception as e:
        print(f"❌ ChromaDB erreur : {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST DES CLÉS API — PROJET P26")
    print("=" * 50)

    test_finnhub()
    test_newsapi()
    test_anthropic()
    test_chromadb()

    print("\n" + "=" * 50)
    print("  Tests terminés.")
    print("=" * 50)