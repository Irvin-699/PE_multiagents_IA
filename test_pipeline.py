"""
test_pipeline.py
Exécute le pipeline complet Agent 1 → 2 → 3 → 4 → 5 et affiche un résumé.
Lance depuis la racine du projet : python test_pipeline.py
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Console Windows en cp1252 : sans cela, le premier caractère non-ASCII
# imprimé (flèches, accents) lève UnicodeEncodeError et tue le pipeline.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

SEP = "=" * 60


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


def check(label: str, value, ok: bool):
    status = "OK" if ok else "ECHEC"
    print(f"  [{status}] {label} : {value}")


# ─── Agent 1 — Données de marché ─────────────────────────────────────────────

section("AGENT 1 — Données de marché")
t0 = time.time()

from agents.market_data import market_data_node

state = {}
try:
    state.update(market_data_node(state))
    md = state["market_data"]
    check("Actifs chargés",  len(md["tickers"]),          len(md["tickers"]) == 10)
    check("Période",         f"{md['start_date']} → {md['end_date']}", True)
    check("Nb jours",        len(md["prices"]),            len(md["prices"]) > 100)
    check("NaN résiduels",   md["prices"].isna().sum().sum(), md["prices"].isna().sum().sum() == 0)
    print(f"  Durée : {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  [ECHEC CRITIQUE] Agent 1 : {e}")
    print("  → Le reste du pipeline ne peut pas tourner sans les données.")
    sys.exit(1)


# ─── Agent 4 — Analyse technique ─────────────────────────────────────────────

section("AGENT 4 — Analyse technique de risque")
t0 = time.time()

from agents.technical import technical_node

try:
    state.update(technical_node(state))
    tr = state["technical_risk"]
    check("Méthode covariance", tr["covariance_method"],      True)
    check("Nb actifs analysés", len(tr["tickers"]),           len(tr["tickers"]) > 0)
    check("Sharpe disponibles", len(tr["sharpe_ratios"]),     len(tr["sharpe_ratios"]) > 0)
    check("Observations",       tr["n_observations"],         tr["n_observations"] > 12)

    print("\n  Volatilités annualisées :")
    for ticker, vol in tr["volatilities"].items():
        print(f"    {ticker:<26} {vol:.1%}")

    print(f"\n  Durée : {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  [ECHEC] Agent 4 : {e}")
    state["technical_risk"] = {}


# ─── Agent 2 — Analyse macro-économique ──────────────────────────────────────

section("AGENT 2 — Analyse macro-économique (1 ETF de test)")
t0 = time.time()

from agents.fundamental import analyze_etf, fundamental_node

# Test sur 1 seul ETF pour limiter la consommation API
TEST_ETF = "Gold"
try:
    result = analyze_etf(TEST_ETF)
    check("Articles collectés",  result["n_articles"],        result["n_articles"] > 0)
    check("Score extrait",       f"{result['score']:+.2f}",   result["score"] != 0.0 or result["n_articles"] == 0)
    check("Analyse non vide",    len(result["analysis"]) > 20, True)

    print(f"\n  Extrait analyse {TEST_ETF} :")
    print("  " + result["analysis"][:200].replace("\n", "\n  ") + "...")

    # Exécute ensuite sur tous les ETFs pour remplir l'état
    print(f"\n  Analyse de tous les ETFs (peut prendre 1-2 min)...")
    state["assets"] = list(state["market_data"]["tickers"])
    state["user_profile"] = {"profil": "Équilibré", "horizon": "medium",
                             "risk_score": 5, "goal": "Croissance équilibrée"}
    state.update(fundamental_node(state))
    check("macro_context dans state", "macro_context" in state, True)
    print(f"  Durée : {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  [ECHEC] Agent 2 : {e}")
    state["macro_context"] = "Analyse macro non disponible."


# ─── Agent 3 — Analyse sentimentale ──────────────────────────────────────────

section("AGENT 3 — Analyse sentimentale")
t0 = time.time()

from agents.sentiment import sentiment_node

try:
    state.update(sentiment_node(state))
    ss = state["sentiment_score"]
    check("Scores disponibles",   len(ss.get("scores", {})),  len(ss.get("scores", {})) > 0)
    check("Score moyen",          f"{ss.get('average_score', 0):+.2f}", True)

    print("\n  Scores sentiment par ETF :")
    for etf, score in ss.get("scores", {}).items():
        bar = "+" * int(abs(score) * 10) if score > 0 else "-" * int(abs(score) * 10)
        print(f"    {etf:<26} {score:+.2f}  {bar}")

    print(f"\n  Durée : {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  [ECHEC] Agent 3 : {e}")
    state["sentiment_score"] = {}


# ─── Agent 5 — Portfolio Manager ─────────────────────────────────────────────

section("AGENT 5 — Portfolio Manager")
t0 = time.time()

from agents.portfolio import portfolio_node

try:
    state.update(portfolio_node(state))

    alloc = state.get("portfolio_allocation", {})
    total = sum(alloc.values())
    check("Allocation générée",   len(alloc),          len(alloc) > 0)
    check("Somme = 100%",         f"{total:.1f}%",     abs(total - 100) < 1.0)

    print("\n  Allocation finale :")
    for asset, pct in sorted(alloc.items(), key=lambda x: -x[1]):
        bar = "#" * int(pct / 2)
        print(f"    {asset:<26} {pct:5.1f}%  {bar}")

    print(f"\n  Extrait recommandation :")
    print("  " + state.get("final_recommendation", "")[:300].replace("\n", "\n  ") + "...")
    print(f"\n  Durée : {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  [ECHEC] Agent 5 : {e}")


# ─── Résumé ───────────────────────────────────────────────────────────────────

section("RESUME DU PIPELINE")
clés_attendues = ["market_data", "technical_risk", "macro_context", "sentiment_score",
                  "portfolio_allocation", "final_recommendation"]
for clé in clés_attendues:
    présent = clé in state and state[clé] not in (None, {}, "")
    print(f"  {'OK' if présent else 'MANQUANT':<8} {clé}")

print()
