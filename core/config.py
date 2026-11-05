"""
core/config.py — Source unique de vérité pour la configuration du projet.

Tout module qui a besoin des tickers, des chemins de données ou des
constantes financières DOIT importer depuis ce fichier. Aucune
redéfinition locale n'est autorisée ailleurs.
"""

from datetime import date
from pathlib import Path

# ─── Chemins (absolus, indépendants du répertoire courant) ────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR     = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "portfolio_daily.parquet"
CSV_PATH     = DATA_DIR / "portfolio_daily.csv"

# Chemin unique de la base vectorielle ChromaDB (partagé Agents 2 & 3)
CHROMA_PATH = str(DATA_DIR / "models" / "chroma")

# ─── Univers d'investissement ─────────────────────────────────────────────────

TICKERS: dict[str, str] = {
    "Nasdaq-100":           "^NDX",      # Invesco    — depuis 1985 · Actions US tech
    "MSCI EM IMI":          "EIMI.L",    # iShares    — depuis 2014 · Actions émergentes
    "STOXX Eur600 Finance": "EXH2.DE",   # iShares    — depuis 2001 · Actions Europe finance
    "Global Healthcare":    "IXJ",       # iShares    — depuis 2001 · Actions santé mondiale
    "US Treasury 10+Y":     "LUTR.L",    # SPDR       — depuis 2016 · Obligations LT US
    "Bund 1-3Y":            "DBXP.DE",   # Xtrackers  — depuis 2007 · Obligations CT Europe
    "Clean Energy":         "ICLN",      # iShares    — depuis 2008 · Énergie propre ESG
    "Global REITs":         "EPRA.PA",   # Amundi     — depuis 2016 · Immobilier monde
    "Gold":                 "GLD",       # SPDR       — depuis 2004 · Or physique
    "Bitcoin":              "BTC-USD",   # Yahoo/CG   — depuis 2014 · Satellite crypto
}

# Début de la période commune (contrainte LUTR.L + EPRA.PA, disponibles depuis 2016)
START_DATE = "2016-01-01"


def end_date() -> str:
    """Fin de période d'analyse : toujours la date du jour."""
    return date.today().isoformat()


# ─── Constantes financières ───────────────────────────────────────────────────

# Taux sans risque annuel (T-Bills US 3 mois)
RF_ANNUAL = 0.043

# ─── Réseau ───────────────────────────────────────────────────────────────────

# Timeout des appels API de données/actualités (secondes)
HTTP_TIMEOUT = 5.0

# Timeout des appels LLM — une complétion Haiku de 400 tokens prend 3 à 8 s,
# un timeout de 5 s la couperait systématiquement
LLM_TIMEOUT = 30.0
