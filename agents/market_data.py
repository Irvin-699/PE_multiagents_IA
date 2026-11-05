"""
agents/market_data.py
Agent 1 — Lecture et mise à jour des données de marché.

Fonctionnement :
    1. Lit portfolio_daily.parquet (créé par data.py)
    2. Télécharge uniquement les jours ouvrés manquants
    3. Met à jour portfolio_daily.parquet et portfolio_daily.csv
    4. Retourne les données depuis 2016-01-01 à LangGraph
"""

import time
import logging

import pandas as pd
import yfinance as yf

from core.state import AgentState
from core.config import TICKERS, START_DATE, PARQUET_PATH, CSV_PATH, HTTP_TIMEOUT

logger = logging.getLogger(__name__)

# Période d'analyse — source unique : core/config.py
ANALYSIS_START = START_DATE


# ─── Fonctions ────────────────────────────────────────────────────────────────

def _strip_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Supprime le timezone de l'index s'il en a un, sans planter s'il n'en a pas."""
    if index.tz is not None:
        return index.tz_convert(None)   # tz_convert supprime proprement un timezone existant
    return index                         # tz_localize(None) planterait si tz déjà présent


def load_prices() -> pd.DataFrame:
    """
    Lit portfolio_daily.parquet généré par data.py.
    Lève une erreur claire si le fichier n'existe pas.
    """
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PARQUET_PATH}\n"
            "Lance d'abord data.py pour générer l'historique."
        )

    prices = pd.read_parquet(PARQUET_PATH)
    prices.index = _strip_timezone(pd.to_datetime(prices.index))
    prices = prices.sort_index()

    logger.info(
        "[Agent 1] Parquet chargé : %d actifs | %d jours | %s → %s",
        prices.shape[1], len(prices),
        prices.index.min().date(), prices.index.max().date(),
    )
    return prices


def update_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Télécharge les jours ouvrés manquants depuis la dernière date du fichier.
    Compare en jours ouvrés pour ne pas lancer de téléchargement inutile
    le weekend ou les jours fériés.
    """
    last_date      = prices.index.max()
    today          = pd.Timestamp.today().normalize()

    # Compter les jours ouvrés manquants (exclut samedi et dimanche)
    business_days_missing = len(pd.bdate_range(start=last_date, end=today)) - 1

    if business_days_missing <= 0:
        print(f"[Agent 1] Données à jour jusqu'au {last_date.date()} — rien à télécharger.")
        return prices

    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    print(f"[Agent 1] {business_days_missing} jour(s) ouvré(s) manquant(s) — mise à jour : {start} → {end}")

    # Téléchargement avec retry et backoff exponentiel
    new_data = None
    for attempt in range(3):
        try:
            raw = yf.download(
                list(TICKERS.values()),
                start       = start,
                end         = end,
                auto_adjust = True,
                progress    = False,
                timeout     = HTTP_TIMEOUT,
            )

            if raw.empty:
                print("[Agent 1] Aucune nouvelle donnée disponible (marché fermé ?).")
                return prices

            # Fix MultiIndex yfinance >= 0.2.x
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]

            raw.index = _strip_timezone(pd.to_datetime(raw.index))

            # Renommer ticker → nom lisible
            ticker_to_name = {v: k for k, v in TICKERS.items()}
            raw = raw.rename(columns=ticker_to_name)

            # Vérifier quels tickers sont revenus
            downloaded  = set(raw.columns)
            expected    = set(TICKERS.keys())
            missing     = expected - downloaded
            if missing:
                logger.warning("[Agent 1] Tickers absents de la mise à jour : %s", missing)

            new_data = raw
            break

        except Exception as e:
            wait = 2 ** attempt   # 1s, 2s, 4s
            logger.warning(
                "[Agent 1] Tentative %d/3 échouée : %s — retry dans %ds.",
                attempt + 1, e, wait,
            )
            time.sleep(wait)

    if new_data is None or new_data.empty:
        logger.error("[Agent 1] Mise à jour échouée — données existantes utilisées.")
        return prices

    # Fusion + suppression des doublons éventuels
    updated = pd.concat([prices, new_data]).sort_index()
    updated = updated[~updated.index.duplicated(keep="last")]

    # Sauvegarde des deux fichiers
    updated.to_parquet(PARQUET_PATH, engine="pyarrow", compression="snappy")
    updated.to_csv(CSV_PATH)
    print(f"[Agent 1] {len(new_data)} nouvelle(s) ligne(s) ajoutée(s) — fichiers mis à jour.")

    return updated


# ─── Nœud LangGraph ──────────────────────────────────────────────────────────

def market_data_node(state: AgentState) -> dict:
    """Agent 1 — Nœud LangGraph."""
    print("--- [Agent 1] Collecte des données ---")

    # 1. Lire le fichier
    prices = load_prices()

    # 2. Mettre à jour si nécessaire
    prices = update_prices(prices)

    # 3. Filtrer sur la période d'analyse + combler les jours fériés
    prices = prices.loc[ANALYSIS_START:].ffill(limit=5)

    print(
        f"[Agent 1] {prices.shape[1]} actifs | {len(prices)} jours | "
        f"{prices.index.min().date()} → {prices.index.max().date()}"
    )

    return {
        "market_data": {
            "prices"    : prices,
            "tickers"   : prices.columns.tolist(),
            "start_date": str(prices.index.min().date()),
            "end_date"  : str(prices.index.max().date()),
        }
    }