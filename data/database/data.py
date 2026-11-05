"""
download_portfolio.py
=====================
Télécharge les données historiques des 10 actifs du portefeuille P26
et les exporte en .parquet et .csv.

Usage :
    python download_portfolio.py

Sorties :
    portfolio_daily.parquet  — données journalières (format compact, rapide à lire)
    portfolio_daily.csv      — même données en CSV lisible
    portfolio_monthly.parquet — version rééchantillonnée en mensuel
    portfolio_monthly.csv

Dépendances :
    pip install yfinance pandas pyarrow
"""

import time
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = {
    "Nasdaq-100":           "^NDX",
    "MSCI EM IMI":          "EIMI.L",
    "STOXX Eur600 Finance": "EXH2.DE",
    "Global Healthcare":    "IXJ",
    "US Treasury 10+Y":     "LUTR.L",
    "Bund 1-3Y":            "DBXP.DE",
    "Clean Energy":         "ICLN",
    "Global REITs":         "EPRA.PA",
    "Gold":                 "GLD",
    "Bitcoin":              "BTC-USD",
}

# Fallbacks si le ticker principal est indisponible
FALLBACKS = {
    "US Treasury 10+Y": ["IDTL.L", "TLT"],
    "Bund 1-3Y":        ["IBGS.L", "EXX3.DE"],
    "Global REITs":     ["VNQ", "REET"],
}

START_DATE  = "2004-01-01"   # On part large — on gardera la période commune ensuite
END_DATE    = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR  = Path(".")       # Dossier de sortie (dossier courant)

# ─── Téléchargement avec retry ────────────────────────────────────────────────

def download_ticker(name: str, ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Télécharge un ticker avec 3 tentatives et backoff exponentiel.
    Gère le bug MultiIndex de yfinance >= 0.2.x.
    """
    candidates = [ticker] + FALLBACKS.get(name, [])
    delay = 1.0

    for t in candidates:
        for attempt in range(1, 4):
            try:
                raw = yf.download(
                    t,
                    start       = start,
                    end         = end,
                    auto_adjust = True,   # Prix ajustés dividendes + splits
                    progress    = False,
                    timeout     = 30,
                )

                if raw is None or raw.empty:
                    raise ValueError("Données vides")

                # Fix MultiIndex yfinance >= 0.2.x
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Garder uniquement Close, renommer en nom lisible
                close = raw[["Close"]].copy()
                close.columns = [name]
                close.index   = pd.to_datetime(close.index).tz_localize(None)

                print(f"  ✅  {name:<26} [{t}]  "
                      f"{close.index.min().date()} → {close.index.max().date()}  "
                      f"({len(close)} jours)")
                return close

            except Exception as e:
                if attempt < 3:
                    print(f"  ⚠   {name} [{t}] tentative {attempt}/3 : {e} — retry {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"  ✗   {name} [{t}] échec définitif")

    return None


# ─── Collecte principale ──────────────────────────────────────────────────────

def collect_all() -> pd.DataFrame:
    """Télécharge tous les tickers et les aligne sur un index commun."""

    print("=" * 65)
    print("  TÉLÉCHARGEMENT DES 10 ACTIFS — PORTEFEUILLE P26")
    print(f"  Période demandée : {START_DATE} → {END_DATE}")
    print("=" * 65)

    frames = []
    for name, ticker in TICKERS.items():
        df = download_ticker(name, ticker, START_DATE, END_DATE)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)   # Petite pause entre les appels pour éviter le rate-limit

    if not frames:
        raise RuntimeError("Aucune donnée téléchargée.")

    # Fusion sur l'index de dates (outer = garder toutes les dates)
    prices = pd.concat(frames, axis=1, join="outer")

    # Forward-fill les jours fériés (max 5 jours consécutifs)
    prices = prices.ffill(limit=5)

    print(f"\n{'─'*65}")
    print(f"  Actifs téléchargés : {prices.shape[1]} / {len(TICKERS)}")
    print(f"  Période totale     : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"  Observations       : {len(prices)} jours")

    # Période commune stricte (sans NaN)
    common = prices.dropna()
    print(f"  Période commune    : {common.index.min().date()} → {common.index.max().date()}")
    print(f"  Obs. communes      : {len(common)} jours")
    print(f"{'─'*65}\n")

    return prices


# ─── Export ──────────────────────────────────────────────────────────────────

def export(prices: pd.DataFrame) -> None:
    """Exporte les données en parquet et CSV — daily et monthly."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Journalier ───────────────────────────────────────────────────────────
    path_parquet = OUTPUT_DIR / "portfolio_daily.parquet"
    path_csv     = OUTPUT_DIR / "portfolio_daily.csv"

    prices.to_parquet(path_parquet, engine="pyarrow", compression="snappy")
    prices.to_csv(path_csv)

    print(f"  💾  {path_parquet}  ({path_parquet.stat().st_size // 1024} Ko)")
    print(f"  💾  {path_csv}  ({path_csv.stat().st_size // 1024} Ko)")

    # ── Mensuel (dernier prix du mois) ────────────────────────────────────────
    freq = "ME" if int(pd.__version__.split(".")[1]) >= 2 else "M"
    monthly = prices.resample(freq).last()

    path_m_parquet = OUTPUT_DIR / "portfolio_monthly.parquet"
    path_m_csv     = OUTPUT_DIR / "portfolio_monthly.csv"

    monthly.to_parquet(path_m_parquet, engine="pyarrow", compression="snappy")
    monthly.to_csv(path_m_csv)

    print(f"  💾  {path_m_parquet}  ({path_m_parquet.stat().st_size // 1024} Ko)")
    print(f"  💾  {path_m_csv}  ({path_m_csv.stat().st_size // 1024} Ko)")

    # ── Résumé des données ────────────────────────────────────────────────────
    print("\n  APERÇU DES DONNÉES (5 premières lignes — daily) :")
    print(prices.head().to_string())
    print("\n  DONNÉES MANQUANTES PAR ACTIF :")
    nan_pct = (prices.isna().mean() * 100).round(1)
    for col, pct in nan_pct.items():
        status = "✓" if pct < 5 else "⚠" if pct < 20 else "✗"
        print(f"  {status}  {col:<26}  {pct:>5.1f}% NaN")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices = collect_all()
    print("EXPORT DES FICHIERS")
    print("=" * 65)
    export(prices)
    print("\n✅  Terminé.")