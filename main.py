"""
main.py — Orchestrateur principal du système Multi-Agents Wisevest (P26)

Architecture d'exécution :
    1. Ordre strict des dépendances :
       Agent 1 → Agent 2 → (Agent 3 ∥ Agent 4) → Agent 5
       L'Agent 3 lit ChromaDB écrit par l'Agent 2 — jamais en parallèle entre eux.
    2. Les boucles INTERNES des Agents 2 et 3 traitent les 10 ETF en
       concurrent (httpx.AsyncClient + AsyncAnthropic + asyncio.gather).
    3. FinBERT chargé une seule fois (singleton core/models.py) ; l'Agent 3
       réutilise les scores stockés dans ChromaDB sans re-scorer.
    4. Timeout global par agent (120s) + fallback gracieux : si un agent
       échoue, le pipeline continue avec un résultat vide.
    5. Chronomètre par agent + timing total en fin de pipeline.

Objectif : pipeline complet < 60s (vs 300-900s en séquentiel).
"""

import sys
import asyncio
import time
import logging
from pathlib import Path

# Console Windows en cp1252 : sans cela, le premier caractère non-ASCII
# imprimé (cadres, accents) lève UnicodeEncodeError et tue le pipeline.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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

# Timeout maximum par agent (secondes) — couvre le premier chargement
# de FinBERT sur machines à faible RAM (le modèle est ensuite en cache)
AGENT_TIMEOUT = 180


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
    print("  PIPELINE MULTI-AGENTS WISEVEST")
    print("═" * 60)

    # Préchauffage FinBERT en tâche de fond : le chargement du modèle
    # (goulot principal sur machine contrainte) recouvre l'Agent 1 et la
    # collecte réseau de l'Agent 2 au lieu de s'ajouter à leur durée.
    loop = asyncio.get_event_loop()
    from core.models import get_finbert
    loop.run_in_executor(None, get_finbert)

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

    ⚠️ IMPORTANT : on n'utilise JAMAIS la boucle asyncio de Streamlit
    (asyncio.get_event_loop() / nest_asyncio). Cette boucle appartient
    au serveur Streamlit et peut être fermée entre deux reruns du script,
    ce qui provoque un crash "RuntimeError: Event loop is closed".

    À la place, on exécute le pipeline dans un THREAD SÉPARÉ avec sa
    PROPRE boucle asyncio isolée, créée et détruite à chaque appel.
    C'est plus sûr et ça fonctionne identiquement en script direct,
    notebook, ou Streamlit.
    """
    import threading

    class AsyncGraph:
        def invoke(self, initial_state: dict) -> dict:
            """
            Lance le pipeline asynchrone dans un thread isolé avec sa
            propre boucle d'événements, puis attend le résultat.
            """
            result_container: dict = {}
            error_container:  list = []

            def _thread_target():
                # Nouvelle boucle asyncio dédiée à ce thread uniquement
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_container["value"] = new_loop.run_until_complete(
                        _run_pipeline(initial_state)
                    )
                except Exception as e:
                    error_container.append(e)
                finally:
                    new_loop.close()

            worker = threading.Thread(target=_thread_target, daemon=True)
            worker.start()
            worker.join()

            if error_container:
                raise error_container[0]

            return result_container.get("value", {})

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