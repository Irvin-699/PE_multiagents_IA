"""
main.py — Orchestrateur principal du système Multi-Agents P26

Optimisations vs version originale :
    1. Agents 2, 3 et 4 tournent en PARALLÈLE après l'Agent 1
       (asyncio.gather — pas de threads, pas de GIL)
    2. Timeout global par agent (120s) — évite les blocages infinis
    3. FinBERT chargé une seule fois et partagé entre Agent 2 et 3
    4. Fallback gracieux : si un agent échoue, le pipeline continue
    5. Chronomètre par agent pour identifier les goulots

Temps estimé : ~80-120s (vs ~900s séquentiel)
"""

import sys
import asyncio
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from core.state import AgentState
from core.config import TICKERS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)

# Timeout maximum par agent (secondes)
AGENT_TIMEOUT = 120


# ─── Wrappers asynchrones ─────────────────────────────────────────────────────

async def _run_agent(name: str, fn, state: dict) -> dict:
    """
    Lance un agent synchrone dans un executor (thread pool) pour ne pas
    bloquer la boucle asyncio pendant les appels réseau.

    Applique un timeout global — si l'agent dépasse AGENT_TIMEOUT secondes,
    il est interrompu et retourne un résultat vide (le pipeline continue).
    """
    loop = asyncio.get_event_loop()
    t0   = time.time()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, fn, state),
            timeout = AGENT_TIMEOUT,
        )
        elapsed = time.time() - t0
        print(f"✅ {name} terminé en {elapsed:.1f}s")
        return result

    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        print(f"⚠️  {name} timeout après {elapsed:.1f}s — résultat vide utilisé")
        return {}

    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ {name} erreur après {elapsed:.1f}s : {e}")
        return {}


# ─── Pipeline asynchrone ──────────────────────────────────────────────────────

async def _run_pipeline(initial_state: dict) -> dict:
    """
    Pipeline des 5 agents avec gestion correcte des dépendances :

        Étape 1 : Agent 1  (collecte — les autres dépendent de ses données)
        Étape 2 : Agent 2  (macro — stocke dans ChromaDB pour l'Agent 3)
        Étape 3 : Agent 3 + Agent 4  (parallèle — Agent 3 lit ChromaDB,
                                       Agent 4 est indépendant)
        Étape 4 : Agent 5  (portfolio — dépend de tout le monde)

    ⚠️ Agent 3 ne peut PAS tourner en parallèle avec Agent 2 :
       il lit ChromaDB que l'Agent 2 est en train d'écrire.
    """
    from agents.market_data import market_data_node
    from agents.fundamental import fundamental_node
    from agents.sentiment   import sentiment_node
    from agents.technical   import technical_node
    from agents.portfolio   import portfolio_node

    state   = dict(initial_state)
    t_start = time.time()

    print("\n" + "═" * 60)
    print("  PIPELINE MULTI-AGENTS P26")
    print("═" * 60)

    # ── Étape 1 : Agent 1 — Collecte des données ─────────────────────────────
    print("\n[1/4] Agent 1 — Collecte des données de marché...")
    result_1 = await _run_agent("Agent 1", market_data_node, state)
    state.update(result_1)

    if not state.get("market_data"):
        print("❌ Agent 1 n'a retourné aucune donnée — abandon du pipeline.")
        return state

    # ── Étape 2 : Agent 2 — Macro (doit finir avant Agent 3) ─────────────────
    print("\n[2/4] Agent 2 — Analyse macro-économique (actualités → ChromaDB)...")
    result_2 = await _run_agent("Agent 2 (Macro)", fundamental_node, state)
    state.update(result_2)

    # ── Étape 3 : Agent 3 + Agent 4 — En PARALLÈLE ───────────────────────────
    # Agent 3 lit ChromaDB (écrit par Agent 2 à l'étape précédente) ✅
    # Agent 4 est indépendant — analyse technique pure ✅
    print("\n[3/4] Agents 3 + 4 — Sentiment & Technique (parallèle)...")
    results_34 = await asyncio.gather(
        _run_agent("Agent 3 (Sentiment)", sentiment_node, state),
        _run_agent("Agent 4 (Technique)", technical_node, state),
        return_exceptions=False,
    )
    for result in results_34:
        if result:
            state.update(result)

    # ── Étape 4 : Agent 5 — Portfolio Manager ────────────────────────────────
    print("\n[4/4] Agent 5 — Optimisation du portefeuille et recommandation...")
    result_5 = await _run_agent("Agent 5 (Portfolio)", portfolio_node, state)
    state.update(result_5)

    t_total = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  Pipeline terminé en {t_total:.1f}s")
    print(f"{'═'*60}\n")

    return state


# ─── Interface publique (appelée par app.py) ──────────────────────────────────

def build_graph():
    """
    Retourne un objet compatible avec l'interface graph.invoke(state)
    attendue par app.py.

    Utilise asyncio pour la parallélisation interne tout en exposant
    une interface synchrone simple.
    """
    class AsyncGraph:
        def invoke(self, initial_state: dict) -> dict:
            """Lance le pipeline asynchrone depuis un contexte synchrone."""
            try:
                # Si une boucle asyncio tourne déjà (ex: Jupyter, Streamlit)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Streamlit tourne dans une boucle — utiliser nest_asyncio
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(_run_pipeline(initial_state))
                else:
                    return loop.run_until_complete(_run_pipeline(initial_state))
            except RuntimeError:
                # Pas de boucle existante — en créer une nouvelle
                return asyncio.run(_run_pipeline(initial_state))

    return AsyncGraph()


# ─── Lancement direct (hors Streamlit) ───────────────────────────────────────

if __name__ == "__main__":
    print("=== Démarrage du système Multi-Agents P26 ===\n")

    initial_state = {
        "assets"              : list(TICKERS.keys()),
        "market_data"         : {},
        "macro_context"       : {},
        "sentiment_score"     : {},
        "technical_risk"      : {},
        "portfolio_allocation": {},
        "final_recommendation": "",
        "user_profile"        : {
            "profil"    : "Équilibré",
            "risk_score": 5,
            "horizon"   : "medium",
            "goal"      : "Équilibrer rendement et risque",
        },
    }

    graph       = build_graph()
    final_state = graph.invoke(initial_state)

    print("\n--- Résultat Final ---")
    print(final_state.get("final_recommendation", "Pas de recommandation."))
    print("\nAllocation :", final_state.get("portfolio_allocation", {}))