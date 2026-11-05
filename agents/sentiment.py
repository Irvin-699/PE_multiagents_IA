"""
agents/sentiment.py
Agent 3 — Analyste Sentimental (version asynchrone)

Rôle dans le pipeline LangGraph :
    Analyse le sentiment des investisseurs pour chaque ETF à partir des
    actualités déjà stockées dans ChromaDB par l'Agent 2.
    Retourne un score sentiment par ETF dans AgentState.sentiment_score.

Ce que l'Agent 3 mesure :
    Pas les faits économiques (c'est l'Agent 2) mais ce que les investisseurs
    PENSENT et RESSENTENT — leurs peurs, espoirs, anticipations et doutes.

Pipeline :
    1. Requête ChromaDB    → récupère les articles de l'Agent 2
                             (avec leurs scores FinBERT déjà calculés —
                             aucun re-scoring, le modèle n'est pas rechargé)
    2. Filtrage sémantique → garde uniquement les articles avec marqueurs
                             de sentiment (verbes d'émotion et de pensée)
    3. Claude Haiku        → synthèse et score final -1 à +1 par ETF
                             (10 ETF en parallèle via AsyncAnthropic)

Dépendance stricte : l'Agent 2 écrit ChromaDB, cet agent le lit.
Ils ne doivent JAMAIS tourner en parallèle entre eux.
"""

import os
import re
import asyncio
import logging

import chromadb
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()   # AVANT tout os.getenv

from core.state import AgentState
from core.config import CHROMA_PATH, LLM_TIMEOUT

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOP_ARTICLES      = 5     # Articles envoyés à Claude Haiku par ETF

# Pondération du score sentiment selon l'horizon d'investissement
# Le sentiment est très peu pertinent à long terme
SENTIMENT_WEIGHT_BY_HORIZON = {
    "short":  0.35,   # Horizon < 2 ans — sentiment très important
    "medium": 0.20,   # Horizon 2–5 ans — sentiment modérément important
    "long":   0.10,   # Horizon 5–10 ans — sentiment peu important
    "vlong":  0.05,   # Horizon > 10 ans — sentiment quasi non pertinent
}

# ─── Marqueurs linguistiques du sentiment investisseur ────────────────────────
#
# Ces expressions capturent deux dimensions :
#   - ÉMOTIONS    : ce que les investisseurs ressentent
#   - COGNITION   : ce que les investisseurs pensent et anticipent

SENTIMENT_MARKERS = [
    # Verbes d'émotion — peur et aversion
    "fear that", "fears that", "worried about", "worry about",
    "concerned about", "panic", "nervous about", "alarmed by",
    "investors flee", "investors dump", "risk aversion",
    "sell-off concerns", "market panic", "traders worry",

    # Verbes d'émotion — optimisme et confiance
    "confident that", "optimistic about", "excited about",
    "enthusiasm", "bullish on", "investors bet on",
    "confidence grows", "optimism returns", "relief",

    # Verbes d'émotion — incertitude et hésitation
    "uncertain about", "doubts about", "hesitate",
    "mixed feelings", "wait and see", "unclear",
    "skeptical about", "cautious about",

    # Verbes cognitifs — anticipation et croyance
    "expect that", "expects that", "anticipate", "anticipates",
    "believe that", "believes that", "predict that", "predicts",
    "hope for", "hopes for", "fear that", "think that",
    "estimate that", "forecast that",

    # Expressions collectives des investisseurs
    "investors expect", "investors believe", "investors fear",
    "investors hope", "market believes", "market expects",
    "traders anticipate", "analysts predict", "analysts warn",
    "sentiment", "mood", "outlook", "perception",
]


# ─── 1. Récupération des articles depuis ChromaDB ────────────────────────────

def _query_chromadb(etf_name: str, n_results: int = 30) -> list[dict]:
    """
    Récupère les articles de l'ETF depuis ChromaDB (alimenté par l'Agent 2),
    avec leurs scores FinBERT déjà présents dans les métadonnées.
    """
    try:
        client     = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("financial_news")

        if collection.count() == 0:
            return []

        # Requête sémantique — articles proches du "sentiment investisseur"
        query = f"investor sentiment fear hope worry expect {etf_name}"

        results = collection.query(
            query_texts = [query],
            n_results   = min(n_results, collection.count()),
            where       = {"etf": etf_name},  # Filtrer par ETF
        )

        articles = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                articles.append({
                    "title"         : doc,
                    "source"        : meta.get("source", ""),
                    "date"          : meta.get("date", ""),
                    "finbert_label" : meta.get("finbert_label", ""),
                    "finbert_score" : float(meta.get("finbert_score", 0)),
                    "url"           : meta.get("url", ""),
                })

        logger.info("[Agent 3] %s : %d articles récupérés depuis ChromaDB",
                    etf_name, len(articles))
        return articles

    except Exception as e:
        logger.warning("[Agent 3] ChromaDB erreur pour %s : %s", etf_name, e)
        return []


# ─── 2. Filtrage sémantique par marqueurs de sentiment ───────────────────────

def _filter_sentiment_articles(articles: list[dict]) -> list[dict]:
    """
    Garde les articles avec marqueurs de sentiment.
    Fallback : si moins de 2 articles matchent, retourne les 3 meilleurs
    par score FinBERT absolu pour ne jamais retourner vide.
    """
    filtered = []
    for article in articles:
        title_lower = article["title"].lower()
        if any(marker in title_lower for marker in SENTIMENT_MARKERS):
            filtered.append(article)

    logger.info("[Agent 3] %d articles conservés après filtrage sémantique "
                "(sur %d)", len(filtered), len(articles))

    # Fallback : si trop peu d'articles matchent les marqueurs stricts,
    # on prend les articles les plus chargés émotionnellement selon FinBERT
    if len(filtered) < 2 and articles:
        logger.info("[Agent 3] Fallback — utilisation des meilleurs articles FinBERT")
        sorted_by_finbert = sorted(
            articles,
            key=lambda x: abs(float(x.get("finbert_score", 0))),
            reverse=True,
        )
        return sorted_by_finbert[:3]

    return filtered


def _rank_by_stored_scores(articles: list[dict]) -> list[dict]:
    """
    Trie les articles par score FinBERT stocké dans ChromaDB (valeur absolue
    décroissante — signaux les plus forts en premier).

    Les scores ont été calculés par l'Agent 2 sur ces mêmes titres :
    les re-scorer ici rechargerait le modèle pour un résultat identique.
    """
    ranked = [
        {
            **article,
            "sentiment_label": article.get("finbert_label", "neutral"),
            "sentiment_score": round(float(article.get("finbert_score", 0)), 4),
        }
        for article in articles
    ]
    ranked.sort(key=lambda x: abs(x["sentiment_score"]), reverse=True)
    return ranked


# ─── 3. Analyse Claude Haiku (asynchrone) ────────────────────────────────────

async def _analyze_with_haiku(anthropic: AsyncAnthropic | None,
                              etf_name: str, articles: list[dict]) -> str:
    """
    Synthèse Claude Haiku sur les articles sentimentaux.

    Le prompt est orienté exclusivement sur les émotions et pensées
    des investisseurs — pas sur les faits économiques.
    """
    if anthropic is None:
        return (f"Analyse sentiment indisponible pour {etf_name} : "
                "ANTHROPIC_API_KEY absente du fichier .env.")

    if not articles:
        return f"Aucun article sentimental disponible pour {etf_name}."

    top      = articles[:TOP_ARTICLES]
    art_text = "\n".join(
        f"- [{a['date']}] {a['title']} "
        f"(FinBERT: {a.get('sentiment_label','?')}, "
        f"score: {a.get('sentiment_score', 0):.2f})"
        for a in top
    )

    prompt = f"""Tu es un analyste spécialisé dans la psychologie des marchés financiers.

Analyse UNIQUEMENT le sentiment et les émotions des investisseurs dans ces titres d'actualités concernant l'ETF {etf_name}. Ne commente pas les faits économiques — concentre-toi sur ce que les investisseurs PENSENT et RESSENTENT.

Actualités sélectionnées pour leur contenu sentimental :
{art_text}

Réponds en français avec ce format exact :

HUMEUR DES INVESTISSEURS :
[2-3 phrases décrivant l'état émotionnel dominant des investisseurs — peur, optimisme, incertitude, euphorie, panique, confiance...]

ANTICIPATIONS COLLECTIVES :
[2-3 phrases sur ce que les investisseurs anticipent et espèrent ou craignent pour cet ETF]

SCORE SENTIMENT : [nombre entre -1.0 et +1.0]
(-1.0 = panique/pessimisme extrême, 0 = sentiment neutre, +1.0 = euphorie/optimisme extrême)

JUSTIFICATION :
[1-2 phrases justifiant le score en termes d'émotions et non de faits]"""

    try:
        response = await anthropic.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error("[Agent 3] Claude Haiku erreur pour %s : %s", etf_name, e)
        return f"Analyse échouée pour {etf_name} : {e}"


def _extract_score(analysis_text: str) -> float:
    """Extrait le score numérique depuis la réponse de Claude Haiku."""
    match = re.search(r"SCORE SENTIMENT\s*:\s*([+-]?\d+\.?\d*)", analysis_text)
    if match:
        score = float(match.group(1))
        return max(-1.0, min(1.0, score))
    return 0.0


# ─── Pipeline asynchrone complet ──────────────────────────────────────────────

async def _analyze_all_sentiments(assets: list[str]) -> dict[str, dict]:
    """
    Pipeline complet pour tous les ETF :
        ChromaDB (I/O, hors event loop) → filtrage → tri par scores stockés
        → Haiku en parallèle (asyncio.gather)
    """
    # Étapes 1-2 — Lecture ChromaDB + filtrage (par ETF, hors event loop)
    prepared: dict[str, list[dict]] = {}
    for etf in assets:
        articles = await asyncio.to_thread(_query_chromadb, etf)
        if not articles:
            logger.warning("[Agent 3] %s : aucun article dans ChromaDB. "
                           "L'Agent 2 doit tourner en premier.", etf)
            prepared[etf] = []
            continue
        sentiment_articles = _filter_sentiment_articles(articles)
        prepared[etf] = _rank_by_stored_scores(sentiment_articles)

    # Étape 3 — Analyses Haiku en parallèle
    anthropic = None
    if ANTHROPIC_API_KEY:
        anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT)
    else:
        logger.warning("[Agent 3] ANTHROPIC_API_KEY absente du .env — "
                       "les analyses LLM seront vides.")

    etfs_with_articles = [etf for etf in assets if prepared[etf]]
    analyses = await asyncio.gather(
        *[_analyze_with_haiku(anthropic, etf, prepared[etf])
          for etf in etfs_with_articles],
        return_exceptions=True,
    )
    analysis_by_etf = dict(zip(etfs_with_articles, analyses))

    results = {}
    for etf in assets:
        articles = prepared[etf]
        if not articles:
            results[etf] = {
                "score"     : 0.0,
                "analysis"  : "Aucun article disponible — lancer l'Agent 2 d'abord.",
                "n_articles": 0,
            }
            continue

        analysis = analysis_by_etf.get(etf, "")
        if isinstance(analysis, Exception):
            logger.error("[Agent 3] Analyse %s exception : %s", etf, analysis)
            analysis = f"Analyse échouée pour {etf} : {analysis}"

        score = _extract_score(analysis)
        logger.info("[Agent 3] %s → score sentiment : %.2f", etf, score)
        results[etf] = {
            "score"     : score,
            "analysis"  : analysis,
            "n_articles": len(articles),
            "top_titles": [a["title"] for a in articles[:TOP_ARTICLES]],
        }
    return results


def analyze_sentiment(etf_name: str) -> dict:
    """Pipeline complet pour UN ETF (compatibilité tests)."""
    return asyncio.run(_analyze_all_sentiments([etf_name]))[etf_name]


# ─── Nœud LangGraph ──────────────────────────────────────────────────────────

def sentiment_node(state: AgentState) -> dict:
    """Agent 3 — Nœud LangGraph (synchrone, boucle asyncio interne)."""
    print("--- [Agent 3] Analyse sentimentale ---")

    assets       = state.get("assets") or []
    user_profile = state.get("user_profile", {})
    horizon      = user_profile.get("horizon", "medium")
    sent_weight  = SENTIMENT_WEIGHT_BY_HORIZON.get(horizon, 0.20)

    # Les 10 ETF sont traités en concurrent (voir note dans fundamental.py)
    try:
        results = asyncio.run(_analyze_all_sentiments(assets))
    except Exception as e:
        logger.error("[Agent 3] Pipeline sentiment échoué : %s", e)
        results = {}

    # Scores et pondération par horizon
    scores   = {k: v["score"] for k, v in results.items()}
    avg      = sum(scores.values()) / len(scores) if scores else 0.0
    weighted = {k: round(v * sent_weight, 4) for k, v in scores.items()}

    print(f"[Agent 3] Score sentiment moyen : {avg:.2f} | "
          f"Poids horizon ({horizon}) : {sent_weight}")
    for etf, score in scores.items():
        tag = "[+]" if score > 0.2 else "[-]" if score < -0.2 else "[=]"
        print(f"  {tag} {etf:<26} {score:+.2f}")

    return {
        "sentiment_score": {
            "scores"          : scores,
            "weighted_scores" : weighted,
            "sentiment_weight": sent_weight,
            "average_score"   : round(avg, 4),
            "analyses"        : {k: v["analysis"] for k, v in results.items()},
            "n_articles"      : {k: v["n_articles"] for k, v in results.items()},
            "horizon"         : horizon,
        }
    }
