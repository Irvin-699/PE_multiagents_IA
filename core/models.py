"""
core/models.py — Modèles ML partagés entre agents.

FinBERT était auparavant chargé deux fois (une instance dans fundamental.py,
une autre dans sentiment.py) : ~2 × 440 Mo de RAM et ~2 × 10 s de chargement.
Ce module expose un singleton unique, thread-safe, chargé paresseusement.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_finbert = None
_lock = threading.Lock()


def get_finbert():
    """
    Retourne l'unique instance FinBERT du processus (chargement paresseux).

    Thread-safe : les agents tournent dans des threads d'executor
    différents — le verrou garantit un seul chargement.
    """
    global _finbert
    if _finbert is None:
        with _lock:
            if _finbert is None:   # double-checked locking
                logger.info("[models] Chargement de FinBERT (unique)...")
                from transformers import pipeline
                _finbert = pipeline(
                    "text-classification",
                    model        = "ProsusAI/finbert",
                    device       = -1,
                    truncation   = True,
                    max_length   = 512,
                    # Machine à 4 Go de RAM : réduit le pic mémoire au chargement
                    model_kwargs = {"low_cpu_mem_usage": True},
                )
                logger.info("[models] FinBERT chargé.")
    return _finbert


def classify_titles(titles: list[str]) -> list[dict]:
    """
    Classifie une liste de titres avec FinBERT.
    Retourne un dict {"label": str, "score": float} par titre.

    L'inférence est sérialisée par le verrou : le pipeline HuggingFace
    ne garantit pas la thread-safety de son état interne.
    """
    if not titles:
        return []
    finbert = get_finbert()
    with _lock:
        return finbert(titles, batch_size=16)
