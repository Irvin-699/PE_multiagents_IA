"""
agents/fundamental.py
Agent 2 — Analyste Macro-Économique (version asynchrone)

Pipeline :
    1. Collecte    → Finnhub (1 seul appel global) + NewsAPI (10 appels parallèles)
    2. Filtrage    → FinBERT (un seul batch pour tous les ETF, modèle partagé)
    3. Stockage    → ChromaDB (vectorisation + persistance — relu par l'Agent 3)
    4. Analyse     → Claude Haiku (10 appels parallèles via AsyncAnthropic)
    5. Score       → Score macro -1 à +1 par ETF → AgentState

Parallélisation :
    Les 10 ETF sont traités en concurrent (asyncio.gather + httpx.AsyncClient).
    Le nœud LangGraph reste synchrone : il crée sa propre boucle asyncio via
    asyncio.run(), ce qui est sûr car main.py l'exécute dans un thread
    d'executor qui ne possède pas de boucle.
"""

import os
import re
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta

import httpx
import chromadb
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()   # AVANT tout os.getenv

from core.state import AgentState
from core.config import CHROMA_PATH, HTTP_TIMEOUT, LLM_TIMEOUT
from core.models import classify_titles

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

MAX_ARTICLES_PER_ETF = 20   # Articles conservés avant filtrage FinBERT
TOP_ARTICLES_FOR_LLM = 5    # Articles envoyés à Claude Haiku


# ─── 1. Collecte des actualités (asynchrone) ─────────────────────────────────

async def _fetch_finnhub_general(client: httpx.AsyncClient,
                                 days_back: int = 7) -> list[dict]:
    """
    Récupère le flux d'actualités générales Finnhub UNE SEULE FOIS.

    L'endpoint /news?category=general renvoie le même corpus quel que soit
    l'ETF — l'ancien code l'appelait 10 fois avec un sleep de 0.5s entre
    chaque appel. Un appel unique, filtré ensuite par mots-clés, suffit.
    """
    if not FINNHUB_API_KEY:
        logger.warning("[Agent 2] FINNHUB_API_KEY absente du .env — Finnhub ignoré.")
        return []

    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")

    try:
        response = await client.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "from": date_from,
                    "to": date_to, "token": FINNHUB_API_KEY},
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning("[Agent 2] Finnhub erreur : %s", e)
        return []


def _filter_finnhub(raw_articles: list[dict], keyword: str) -> list[dict]:
    """Filtre le corpus Finnhub global sur un mot-clé (titre ou résumé)."""
    keyword_lower = keyword.lower()
    filtered = [
        {
            "title" : a.get("headline", ""),
            "source": "finnhub",
            "date"  : datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
            "url"   : a.get("url", ""),
        }
        for a in raw_articles
        if keyword_lower in a.get("headline", "").lower()
        or keyword_lower in a.get("summary", "").lower()
    ]
    return filtered[:MAX_ARTICLES_PER_ETF]


async def _fetch_newsapi(client: httpx.AsyncClient, keyword: str) -> list[dict]:
    """
    Collecte les actualités depuis NewsAPI (plan gratuit : 100 requêtes/jour).
    Clé gratuite sur newsapi.org → .env : NEWSAPI_KEY=ta_cle
    """
    if not NEWSAPI_KEY:
        logger.warning("[Agent 2] NEWSAPI_KEY absente du .env — NewsAPI ignoré.")
        return []

    try:
        response = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q"       : f"{keyword} finance economy",
                "language": "en",
                "sortBy"  : "publishedAt",
                "pageSize": 10,
                "from"    : (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "apiKey"  : NEWSAPI_KEY,
            },
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


def _dedupe(articles: list[dict]) -> list[dict]:
    """Déduplique par empreinte des 50 premiers caractères du titre."""
    seen, dedup = set(), []
    for article in articles:
        key = hashlib.md5(article["title"][:50].encode()).hexdigest()
        if key not in seen and article["title"]:
            seen.add(key)
            dedup.append(article)
    return dedup


async def _collect_all_news(assets: list[str]) -> dict[str, list[dict]]:
    """
    Collecte les actualités de TOUS les ETF en parallèle :
    1 appel Finnhub global + 1 appel NewsAPI par ETF, tous concurrents.
    """
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        finnhub_task  = _fetch_finnhub_general(client)
        newsapi_tasks = [
            _fetch_newsapi(client, ETF_KEYWORDS.get(etf, [etf])[0])
            for etf in assets
        ]
        finnhub_raw, *newsapi_results = await asyncio.gather(
            finnhub_task, *newsapi_tasks, return_exceptions=True,
        )

    if isinstance(finnhub_raw, Exception):
        logger.warning("[Agent 2] Finnhub exception : %s", finnhub_raw)
        finnhub_raw = []

    news_by_etf = {}
    for etf, newsapi_articles in zip(assets, newsapi_results):
        if isinstance(newsapi_articles, Exception):
            logger.warning("[Agent 2] NewsAPI exception pour %s : %s", etf, newsapi_articles)
            newsapi_articles = []
        keyword  = ETF_KEYWORDS.get(etf, [etf])[0]
        articles = _filter_finnhub(finnhub_raw, keyword) + newsapi_articles
        news_by_etf[etf] = _dedupe(articles)
        logger.info("[Agent 2] %s : %d articles collectés", etf, len(news_by_etf[etf]))

    return news_by_etf


# ─── 2. Filtrage FinBERT (un seul batch global) ──────────────────────────────

def _score_all_with_finbert(news_by_etf: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    Score tous les articles de tous les ETF en UN SEUL batch FinBERT
    (modèle partagé core/models.py), puis trie chaque ETF par signal décroissant.
    Neutre → score réduit (×0.1 de la confiance) comme dans l'ancienne version.
    """
    flat: list[tuple[str, dict]] = [
        (etf, article) for etf, articles in news_by_etf.items() for article in articles
    ]
    if not flat:
        return {etf: [] for etf in news_by_etf}

    titles = [article["title"] for _, article in flat]
    try:
        results = classify_titles(titles)
    except Exception as e:
        logger.warning("[Agent 2] FinBERT erreur : %s — articles non scorés.", e)
        return news_by_etf

    scored_by_etf: dict[str, list[dict]] = {etf: [] for etf in news_by_etf}
    for (etf, article), result in zip(flat, results):
        label = result["label"].lower()
        conf  = result["score"]
        if label == "positive":
            num_score = conf
        elif label == "negative":
            num_score = -conf
        else:
            num_score = conf * 0.1
        scored_by_etf[etf].append({
            **article,
            "finbert_label": label,
            "finbert_score": round(num_score, 4),
        })

    for etf in scored_by_etf:
        scored_by_etf[etf].sort(key=lambda x: abs(x.get("finbert_score", 0)), reverse=True)
    return scored_by_etf


# ─── 3. Stockage ChromaDB ────────────────────────────────────────────────────

def store_in_chromadb(etf_name: str, articles: list[dict],
                      chroma_path: str = CHROMA_PATH) -> None:
    """
    Stocke les articles dans ChromaDB (chemin absolu unique de core/config.py).
    L'Agent 3 relit cette collection — l'Agent 2 doit donc finir avant lui.
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


# ─── 4. Analyse Claude Haiku (asynchrone) ────────────────────────────────────

async def _analyze_with_haiku(anthropic: AsyncAnthropic | None,
                              etf_name: str, articles: list[dict]) -> str:
    """Analyse les meilleurs articles avec Claude Haiku (appel async)."""
    if anthropic is None:
        return (f"Analyse indisponible pour {etf_name} : "
                "ANTHROPIC_API_KEY absente du fichier .env.")

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
        response = await anthropic.messages.create(
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
    match = re.search(r"SCORE MACRO\s*:\s*([+-]?\d+\.?\d*)", analysis_text)
    if match:
        score = float(match.group(1))
        return max(-1.0, min(1.0, score))
    return 0.0


# ─── 5. Pipeline asynchrone complet ──────────────────────────────────────────

async def _analyze_all_etfs(assets: list[str]) -> dict[str, dict]:
    """
    Pipeline complet pour tous les ETF, entièrement parallélisé :
        collecte (async) → FinBERT (1 batch) → ChromaDB → Haiku (async gather)
    """
    # Étapes 1-2 — Collecte parallèle puis scoring FinBERT global.
    # FinBERT est CPU-bound : on le sort de la boucle d'événements.
    news_by_etf   = await _collect_all_news(assets)
    scored_by_etf = await asyncio.to_thread(_score_all_with_finbert, news_by_etf)

    # Étape 3 — Stockage ChromaDB (I/O disque, hors event loop)
    for etf, articles in scored_by_etf.items():
        if articles:
            await asyncio.to_thread(store_in_chromadb, etf, articles)

    # Étape 4 — Analyses Haiku en parallèle
    anthropic = None
    if ANTHROPIC_API_KEY:
        anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT)
    else:
        logger.warning("[Agent 2] ANTHROPIC_API_KEY absente du .env — "
                       "les analyses LLM seront vides.")

    analyses = await asyncio.gather(
        *[_analyze_with_haiku(anthropic, etf, scored_by_etf.get(etf, []))
          for etf in assets],
        return_exceptions=True,
    )

    results = {}
    for etf, analysis in zip(assets, analyses):
        if isinstance(analysis, Exception):
            logger.error("[Agent 2] Analyse %s exception : %s", etf, analysis)
            analysis = f"Analyse échouée pour {etf} : {analysis}"
        articles = scored_by_etf.get(etf, [])
        score    = _extract_score(analysis)
        logger.info("[Agent 2] %s → score macro : %.2f", etf, score)
        results[etf] = {
            "score"     : score,
            "analysis"  : analysis,
            "n_articles": len(articles),
            "top_titles": [a["title"] for a in articles[:TOP_ARTICLES_FOR_LLM]],
        }
    return results


def analyze_etf(etf_name: str) -> dict:
    """
    Pipeline complet pour UN ETF (compatibilité test_pipeline.py).
    """
    return asyncio.run(_analyze_all_etfs([etf_name]))[etf_name]


# ─── Nœud LangGraph ──────────────────────────────────────────────────────────

def fundamental_node(state: AgentState) -> dict:
    """Agent 2 — Nœud LangGraph (synchrone, boucle asyncio interne)."""
    print("--- [Agent 2] Analyse macro-économique ---")

    assets       = state.get("assets") or list(ETF_KEYWORDS.keys())
    user_profile = state.get("user_profile", {})
    horizon      = user_profile.get("horizon", "medium")
    macro_weight = MACRO_WEIGHT_BY_HORIZON.get(horizon, 0.25)

    # Les 10 ETF sont traités en concurrent. asyncio.run est sûr ici :
    # ce nœud tourne dans un thread d'executor sans boucle existante.
    try:
        results = asyncio.run(_analyze_all_etfs(assets))
    except Exception as e:
        logger.error("[Agent 2] Pipeline macro échoué : %s", e)
        results = {}

    # Scores et pondération
    scores   = {k: v["score"] for k, v in results.items()}
    avg      = sum(scores.values()) / len(scores) if scores else 0.0
    weighted = {k: round(v * macro_weight, 4) for k, v in scores.items()}

    print(f"[Agent 2] Score macro moyen : {avg:.2f} | Poids horizon ({horizon}) : {macro_weight}")
    for etf, score in scores.items():
        tag = "[+]" if score > 0.2 else "[-]" if score < -0.2 else "[=]"
        print(f"  {tag} {etf:<26} {score:+.2f}")

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
