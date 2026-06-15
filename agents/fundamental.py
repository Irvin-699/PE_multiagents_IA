"""
agents/fundamental.py
Agent 2 — Analyste Macro-Économique

Pipeline :
    1. Collecte    → Finnhub + NewsAPI (actualités brutes)
    2. Filtrage    → FinBERT (score sentiment rapide sur tous les articles)
    3. Stockage    → ChromaDB (vectorisation + persistance)
    4. Analyse     → Claude Haiku (analyse des 5 meilleurs articles)
    5. Score       → Score macro -1 à +1 par ETF → AgentState

LLM : Claude 3.5 Haiku (~5€/mois)
"""

import os
import time
import logging
import hashlib
from datetime import datetime, timedelta

import requests
import chromadb
from anthropic import Anthropic
from transformers import pipeline
from dotenv import load_dotenv

from core.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY")
NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Mots-clés de recherche par ETF
ETF_KEYWORDS = {
    "Nasdaq-100":           ["Fed rate tech nasdaq"],
    "MSCI EM IMI":          ["emerging markets economy"],
    "STOXX Eur600 Finance": ["ECB interest rates"],
    "Global Healthcare":    ["FDA approval"],
    "US Treasury 10+Y":     ["US inflation CPI"],
    "Bund 1-3Y":            ["eurozone inflation"],
    "Clean Energy":         ["renewable energy policy"],
    "Global REITs":         ["REIT earnings"],
    "Gold":                 ["gold inflation hedge"],
    "Bitcoin":              ["Bitcoin crypto market"],
}

# Pondération du score macro selon l'horizon d'investissement
MACRO_WEIGHT_BY_HORIZON = {
    "short":  0.40,
    "medium": 0.25,
    "long":   0.15,
    "vlong":  0.05,
}

MAX_ARTICLES_PER_ETF = 20   # Articles collectés avant filtrage FinBERT
TOP_ARTICLES_FOR_LLM = 5    # Articles envoyés à Claude Haiku
FINNHUB_DELAY        = 0.5  # Délai entre appels Finnhub (rate limit)


# ─── 1. Collecte des actualités ───────────────────────────────────────────────

def _fetch_finnhub(keyword: str, days_back: int = 7) -> list[dict]:
    """
    Collecte les actualités depuis Finnhub.
    Plan gratuit : 60 appels/minute.
    """
    if not FINNHUB_API_KEY:
        logger.warning("[Agent 2] Clé Finnhub manquante.")
        return []

    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/news",
            params  = {"category": "general", "from": date_from,
                       "to": date_to, "token": FINNHUB_API_KEY},
            timeout = 10,
        )
        response.raise_for_status()
        articles = response.json()

        keyword_lower = keyword.lower()
        filtered = [
            {
                "title" : a.get("headline", ""),
                "source": "finnhub",
                "date"  : datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
                "url"   : a.get("url", ""),
            }
            for a in articles
            if keyword_lower in a.get("headline", "").lower()
            or keyword_lower in a.get("summary", "").lower()
        ]

        time.sleep(FINNHUB_DELAY)
        return filtered[:MAX_ARTICLES_PER_ETF]

    except Exception as e:
        logger.warning("[Agent 2] Finnhub erreur pour '%s' : %s", keyword, e)
        return []


def _fetch_newsapi(keyword: str) -> list[dict]:
    """
    Collecte les actualités depuis NewsAPI.
    Plan gratuit : 100 requêtes/jour — suffisant pour 10 ETFs × 3 keywords.

    Pour obtenir la clé :
        1. Va sur newsapi.org
        2. Clique sur 'Get API Key'
        3. Crée un compte gratuit
        4. Copie la clé dans ton .env : NEWSAPI_KEY=ta_cle
    """
    if not NEWSAPI_KEY:
        logger.warning("[Agent 2] Clé NewsAPI manquante.")
        return []

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params  = {
                "q"           : f"{keyword} finance economy",
                "language"    : "en",
                "sortBy"      : "publishedAt",
                "pageSize"    : 10,
                "from"        : (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "apiKey"      : NEWSAPI_KEY,
            },
            timeout = 10,
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])

        return [
            {
                "title" : a.get("title", ""),
                "source": a.get("source", {}).get("name", "newsapi"),
                "date"  : a.get("publishedAt", "")[:10],
                "url"   : a.get("url", ""),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]

    except Exception as e:
        logger.warning("[Agent 2] NewsAPI erreur pour '%s' : %s", keyword, e)
        return []


def collect_news(etf_name: str) -> list[dict]:
    """
    Collecte et déduplique les actualités depuis Finnhub + NewsAPI.
    """
    keywords    = ETF_KEYWORDS.get(etf_name, [etf_name])
    all_articles = []

    for keyword in keywords:
        all_articles.extend(_fetch_finnhub(keyword))
        all_articles.extend(_fetch_newsapi(keyword))

    # Dédupliquer par titre
    seen, dedup = set(), []
    for article in all_articles:
        key = hashlib.md5(article["title"][:50].encode()).hexdigest()
        if key not in seen and article["title"]:
            seen.add(key)
            dedup.append(article)

    logger.info("[Agent 2] %s : %d articles collectés", etf_name, len(dedup))
    return dedup


# ─── 2. Filtrage FinBERT ──────────────────────────────────────────────────────

_finbert_pipeline = None

def _get_finbert():
    """Charge FinBERT une seule fois (chargement paresseux)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("[Agent 2] Chargement de FinBERT...")
        _finbert_pipeline = pipeline(
            "text-classification",
            model      = "ProsusAI/finbert",
            device     = -1,
            truncation = True,
            max_length = 512,
        )
        logger.info("[Agent 2] FinBERT chargé.")
    return _finbert_pipeline


def score_with_finbert(articles: list[dict]) -> list[dict]:
    """
    Score chaque article avec FinBERT et trie par pertinence décroissante.
    Conversion : positive → +score | negative → -score | neutral → score réduit
    """
    if not articles:
        return []

    finbert = _get_finbert()
    titles  = [a["title"] for a in articles]

    try:
        results = finbert(titles, batch_size=16)
    except Exception as e:
        logger.warning("[Agent 2] FinBERT erreur : %s", e)
        return articles

    scored = []
    for article, result in zip(articles, results):
        label = result["label"].lower()
        conf  = result["score"]

        if label == "positive":
            num_score = conf
        elif label == "negative":
            num_score = -conf
        else:
            num_score = conf * 0.1

        scored.append({
            **article,
            "finbert_label": label,
            "finbert_score": round(num_score, 4),
        })

    # Trier par valeur absolue décroissante
    scored.sort(key=lambda x: abs(x["finbert_score"]), reverse=True)
    return scored


# ─── 3. Stockage ChromaDB ────────────────────────────────────────────────────

def store_in_chromadb(
    etf_name   : str,
    articles   : list[dict],
    chroma_path: str = "./data/models/chroma",
) -> None:
    """
    Stocke les articles dans ChromaDB pour recherche sémantique future.
    Utilisé aussi par l'Agent 3 (sentiment) pour éviter de retélécharger.
    """
    try:
        client     = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name     = "financial_news",
            metadata = {"hnsw:space": "cosine"},
        )

        ids, docs, metas = [], [], []
        for article in articles:
            doc_id = hashlib.md5(article["title"].encode()).hexdigest()
            ids.append(doc_id)
            docs.append(article["title"])
            metas.append({
                "etf"           : etf_name,
                "source"        : article.get("source", ""),
                "date"          : article.get("date", ""),
                "finbert_label" : article.get("finbert_label", ""),
                "finbert_score" : str(article.get("finbert_score", 0)),
                "url"           : article.get("url", ""),
            })

        if ids:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info("[Agent 2] %s : %d articles stockés dans ChromaDB.", etf_name, len(ids))

    except Exception as e:
        logger.warning("[Agent 2] ChromaDB erreur pour %s : %s", etf_name, e)


# ─── 4. Analyse Claude Haiku ─────────────────────────────────────────────────

def analyze_with_haiku(etf_name: str, articles: list[dict]) -> str:
    """
    Analyse les TOP_ARTICLES_FOR_LLM meilleurs articles avec Claude Haiku.
    Retourne une analyse structurée avec un score macro justifié.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("[Agent 2] Clé Anthropic manquante.")
        return f"Analyse indisponible pour {etf_name}."

    if not articles:
        return f"Aucune actualité disponible pour {etf_name}."

    top_articles  = articles[:TOP_ARTICLES_FOR_LLM]
    articles_text = "\n".join(
        f"- [{a['date']}] {a['title']} "
        f"(sentiment: {a.get('finbert_label','?')}, score: {a.get('finbert_score', 0):.2f})"
        for a in top_articles
    )

    prompt = f"""Tu es un analyste macro-économique expert. Analyse l'impact \
des actualités suivantes sur l'ETF {etf_name}.

Actualités récentes (filtrées par FinBERT) :
{articles_text}

Réponds en français avec ce format exact :

CONTEXTE MACRO :
[2-3 phrases résumant le contexte économique actuel pour cet ETF]

IMPACT SUR {etf_name.upper()} :
[2-3 phrases expliquant l'impact spécifique sur cet ETF]

SCORE MACRO : [nombre entre -1.0 et +1.0]
(-1.0 = très défavorable, 0 = neutre, +1.0 = très favorable)

JUSTIFICATION DU SCORE :
[1-2 phrases justifiant le score]"""

    try:
        client   = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    except Exception as e:
        logger.error("[Agent 2] Claude Haiku erreur pour %s : %s", etf_name, e)
        return f"Analyse échouée pour {etf_name} : {e}"


def _extract_score(analysis_text: str) -> float:
    """Extrait le score numérique depuis la réponse de Claude Haiku."""
    import re
    match = re.search(r"SCORE MACRO\s*:\s*([+-]?\d+\.?\d*)", analysis_text)
    if match:
        score = float(match.group(1))
        return max(-1.0, min(1.0, score))
    return 0.0


# ─── 5. Pipeline complet par ETF ─────────────────────────────────────────────

def analyze_etf(etf_name: str) -> dict:
    """
    Pipeline complet pour un ETF :
    collecte → FinBERT → ChromaDB → Haiku → score
    """
    logger.info("[Agent 2] Analyse de %s...", etf_name)

    # Étape 1 — Collecte
    articles = collect_news(etf_name)
    if not articles:
        logger.warning("[Agent 2] %s : aucun article collecté.", etf_name)
        return {"score": 0.0, "analysis": "Aucune actualité disponible.", "n_articles": 0}

    # Étape 2 — Filtrage FinBERT
    scored_articles = score_with_finbert(articles)

    # Étape 3 — Stockage ChromaDB
    store_in_chromadb(etf_name, scored_articles)

    # Étape 4 — Analyse Claude Haiku
    analysis = analyze_with_haiku(etf_name, scored_articles)

    # Étape 5 — Extraction du score
    score = _extract_score(analysis)

    logger.info("[Agent 2] %s → score macro : %.2f", etf_name, score)

    return {
        "score"     : score,
        "analysis"  : analysis,
        "n_articles": len(articles),
        "top_titles": [a["title"] for a in scored_articles[:TOP_ARTICLES_FOR_LLM]],
    }


# ─── Nœud LangGraph ──────────────────────────────────────────────────────────

def fundamental_node(state: AgentState) -> dict:
    """Agent 2 — Nœud LangGraph."""
    print("--- [Agent 2] Analyse macro-économique ---")

    assets       = state.get("assets") or list(ETF_KEYWORDS.keys())
    user_profile = state.get("user_profile", {})
    horizon      = user_profile.get("horizon", "medium")
    macro_weight = MACRO_WEIGHT_BY_HORIZON.get(horizon, 0.25)

    # Analyser chaque ETF
    results = {}
    for etf_name in assets:
        if etf_name in ETF_KEYWORDS:
            results[etf_name] = analyze_etf(etf_name)

    # Scores et pondération
    scores   = {k: v["score"] for k, v in results.items()}
    avg      = sum(scores.values()) / len(scores) if scores else 0.0
    weighted = {k: round(v * macro_weight, 4) for k, v in scores.items()}

    print(f"[Agent 2] Score macro moyen : {avg:.2f} | Poids horizon ({horizon}) : {macro_weight}")
    for etf, score in scores.items():
        emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
        print(f"  {emoji} {etf:<26} {score:+.2f}")

    return {
        "macro_context": {
            "scores"          : scores,
            "weighted_scores" : weighted,
            "macro_weight"    : macro_weight,
            "average_score"   : round(avg, 4),
            "analyses"        : {k: v["analysis"] for k, v in results.items()},
            "n_articles"      : {k: v["n_articles"] for k, v in results.items()},
            "horizon"         : horizon,
        }
    }