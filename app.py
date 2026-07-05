"""
app.py — Interface Streamlit du projet P26
Questionnaire KYC complet (10 questions) + pipeline LangGraph + affichage résultats

Lancement : python -m streamlit run app.py
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

# ── Fix logging : force les messages agents à apparaître dans le terminal ──────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    force  = True,          # écrase la config Streamlit existante
    stream = sys.stdout,    # redirige vers stdout (terminal)
)

# ── Configuration page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAgent P26",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING Q8 : libellés grand-public → clés internes TICKERS
# Les valeurs DOIVENT correspondre exactement à TICKERS.keys() dans
# market_data.py — portfolio.py s'en sert pour filtrer known_assets
# ─────────────────────────────────────────────────────────────────────────────

Q8_LABEL_TO_KEY = {
    "📈 Actions technologiques mondiales (ex. Apple, Microsoft, Google)":
        "Nasdaq-100",
    "🌍 Actions des pays en développement (ex. Inde, Brésil, Chine)":
        "MSCI EM IMI",
    "🏦 Actions du secteur bancaire et financier en Europe":
        "STOXX Eur600 Finance",
    "💊 Actions du secteur médical et pharmaceutique mondial":
        "Global Healthcare",
    "🏛️ Obligations d'État américaines à long terme":
        "US Treasury 10+Y",
    "🇩🇪 Obligations d'État allemandes à court terme":
        "Bund 1-3Y",
    "🌱 Actions d'entreprises dans les énergies renouvelables":
        "Clean Energy",
    "🏢 Investissement immobilier coté en bourse (bureaux, centres commerciaux)":
        "Global REITs",
    "🥇 Or (valeur refuge en période d'incertitude)":
        "Gold",
    "₿ Bitcoin":
        "Bitcoin",
}

Q8_LABELS = list(Q8_LABEL_TO_KEY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DU SCORE KYC
# ─────────────────────────────────────────────────────────────────────────────

def compute_kyc_score(answers: dict) -> tuple[int, str, int, str]:
    """
    Calcule le score total (10-31), le profil et le risk_score (1-10).
    q8 doit déjà contenir des clés internes (ex. "Nasdaq-100"), pas des libellés.
    Retourne : (score, profil_label, risk_score, horizon_code)
    """
    score = 0

    # Q1 — Placement
    score += {"A": 1, "B": 2, "C": 3}[answers["q1"]]

    # Q2 — Réaction à une perte de 20%
    score += {"Vends tout": 1, "J'attends": 2, "Je renforce": 3}[answers["q2"]]

    # Q3 — 50/50 doubler ou perdre 1/3
    score += 1 if answers["q3"] == "Non, je refuse" else 3

    # Q4 — 50/50 gain fort ou perdre la moitié
    score += 1 if answers["q4"] == "Non, je refuse" else 3

    # Q5 — Capacité d'adaptation financière
    score += {"Très peu": 1, "Peu": 2, "Assez grande": 3}[answers["q5"]]

    # Q6 — Horizon d'investissement
    score += {"Moins de 2 ans": 1, "2 à 5 ans": 2, "5 à 10 ans": 3, "Plus de 10 ans": 4}[answers["q6"]]

    # Q7 — Besoin d'argent dans 2 ans
    score += 1 if answers["q7"] == "Oui, probablement" else 3

    # Q8 — Actifs connus (nombre de sélections)
    n = len(answers["q8"])
    score += 1 if n <= 2 else (2 if n <= 6 else 3)

    # Q9 — Importance ESG
    score += {"Aucune": 1, "Modérée": 2, "Forte": 3}[answers["q9"]]

    # Q10 — Objectif principal
    score += {"Protéger mon capital": 1, "Équilibrer rendement et risque": 2,
              "Maximiser le rendement": 3}[answers["q10"]]

    # Profil et risk_score
    if score <= 16:
        profil     = "Prudent"
        risk_score = 1 + round((score - 10) / (16 - 10) * 2)   # 1-3
    elif score <= 23:
        profil     = "Équilibré"
        risk_score = 4 + round((score - 17) / (23 - 17) * 2)   # 4-6
    else:
        profil     = "Dynamique"
        risk_score = 7 + round((score - 24) / (31 - 24) * 3)   # 7-10

    # Horizon pour fundamental.py
    horizon_map = {
        "Moins de 2 ans": "short",
        "2 à 5 ans":      "medium",
        "5 à 10 ans":     "long",
        "Plus de 10 ans": "vlong",
    }
    horizon = horizon_map[answers["q6"]]
    # Si besoin de l'argent dans 2 ans → on force court terme
    if answers["q7"] == "Oui, probablement":
        horizon = "short"

    return score, profil, max(1, min(10, risk_score)), horizon


# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 FinAgent P26 — Analyse Financière Multi-Agents")
st.caption("UTT · Irvin TEMFACK & Ségolène SIEWE · 2026")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT : questionnaire rempli ou résultats disponibles ?
# ─────────────────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    # ── PAGE RÉSULTATS ────────────────────────────────────────────────────────
    profile = st.session_state["kyc_profile"]
    results = st.session_state["results"]

    # Bandeau de profil
    couleur = {"Prudent": "🟢", "Équilibré": "🟡", "Dynamique": "🔴"}
    score   = st.session_state["kyc_score"]
    st.success(
        f"{couleur[profile['profil']]} **Profil : {profile['profil']}** "
        f"· Score KYC : {score}/31 · Tolérance au risque : {profile['risk_score']}/10"
    )

    tab1, tab2, tab3 = st.tabs(
        ["💬 Recommandation", "📊 Allocation & Sentiment", "📈 Risque Technique"]
    )

    # ── Onglet 1 : Recommandation ─────────────────────────────────────────────
    with tab1:
        reco = results.get("final_recommendation", "")
        if reco:
            st.markdown(reco)
        else:
            st.warning("Aucune recommandation — vérifiez votre clé ANTHROPIC_API_KEY.")

    # ── Onglet 2 : Allocation + Sentiment ────────────────────────────────────
    with tab2:
        col_a, col_s = st.columns(2)

        with col_a:
            st.subheader("💼 Allocation proposée")
            alloc = results.get("portfolio_allocation", {})
            if alloc:
                df = (pd.DataFrame({"ETF": list(alloc.keys()),
                                    "Allocation (%)": list(alloc.values())})
                        .sort_values("Allocation (%)", ascending=False))
                st.bar_chart(df.set_index("ETF"), use_container_width=True)
                st.dataframe(df.style.format({"Allocation (%)": "{:.1f}%"}),
                             hide_index=True, use_container_width=True)
            else:
                st.warning("Allocation non disponible.")

        with col_s:
            st.subheader("💭 Sentiment par ETF")
            sent = results.get("sentiment_score", {})
            if isinstance(sent, dict) and "scores" in sent:
                sent = sent["scores"]
            if sent:
                df_s = pd.DataFrame({"ETF": list(sent.keys()), "Sentiment": list(sent.values())})
                st.bar_chart(df_s.set_index("ETF"), use_container_width=True)
            else:
                st.warning("Scores sentiment non disponibles.")

    # ── Onglet 3 : Risque Technique ───────────────────────────────────────────
    with tab3:
        tech = results.get("technical_risk", {})
        if tech:
            vols = tech.get("volatilities", {})
            if vols:
                st.subheader("Volatilité annualisée")
                df_v = pd.DataFrame({"ETF": list(vols.keys()),
                                     "Volatilité (%)": [v * 100 for v in vols.values()]}
                                    ).sort_values("Volatilité (%)", ascending=False)
                st.bar_chart(df_v.set_index("ETF"), use_container_width=True)

            sharpe = tech.get("sharpe_ratios", {})
            if sharpe:
                st.subheader("Ratio de Sharpe")
                df_sh = pd.DataFrame({"ETF": list(sharpe.keys()),
                                      "Sharpe": list(sharpe.values())}
                                     ).sort_values("Sharpe", ascending=False)
                st.dataframe(df_sh.style.format({"Sharpe": "{:.2f}"}),
                             hide_index=True, use_container_width=True)
        else:
            st.warning("Données techniques non disponibles.")

    st.divider()
    if st.button("🔄 Refaire le questionnaire"):
        for k in ["results", "kyc_profile", "kyc_score", "kyc_done"]:
            st.session_state.pop(k, None)
        st.rerun()


elif "kyc_done" in st.session_state:
    # ── LANCEMENT DU PIPELINE ─────────────────────────────────────────────────
    profile = st.session_state["kyc_profile"]
    score   = st.session_state["kyc_score"]

    couleur = {"Prudent": "🟢", "Équilibré": "🟡", "Dynamique": "🔴"}
    st.info(
        f"{couleur[profile['profil']]} **Profil : {profile['profil']}** "
        f"· Score : {score}/31 · Risque : {profile['risk_score']}/10 "
        f"· Horizon : {profile['horizon']}"
    )

    st.subheader("Lancer l'analyse du portefeuille")
    st.markdown(
        "Les 5 agents IA vont maintenant analyser votre portefeuille. "
        "**Cette étape prend 1 à 3 minutes** (FinBERT + APIs financières + Claude Sonnet)."
    )

    if st.button("🚀 Analyser mon portefeuille", type="primary", use_container_width=True):

        # ── Panneau de progression visible dans l'UI ──────────────────────────
        with st.status("🔄 Analyse multi-agents en cours…", expanded=True) as status:

            log_area = st.empty()
            logs: list[str] = []

            def ui_log(msg: str) -> None:
                """Écrit dans le panneau Streamlit ET dans le terminal."""
                logs.append(msg)
                log_area.markdown("\n\n".join(logs))
                print(msg, flush=True)

            try:
                ui_log("⚙️ **Initialisation** — chargement du graphe d'agents…")

                from main import build_graph
                from core.config import TICKERS

                graph = build_graph()

                initial_state = {
                    # ✅ .keys() → noms d'ETF ("Nasdaq-100", …), pas les tickers boursiers
                    "assets": list(TICKERS.keys()),
                    "user_profile": {
                        "profil":       profile["profil"],          # ✅ AJOUT CRITIQUE — Prudent/Équilibré/Dynamique
                        "risk_score":   profile["risk_score"],
                        "horizon":      profile["horizon"],
                        "goal":         profile["goal"],
                        "known_assets": profile.get("known_assets", list(TICKERS.keys())),
                    },
                    "market_data":          {},
                    "macro_context":        {},
                    "sentiment_score":      {},
                    "technical_risk":       {},
                    "portfolio_allocation": {},
                    "final_recommendation": "",
                }

                ui_log("")
                ui_log("---")
                ui_log("📡 — Collecte des cours de marché (yfinance / Binance)…")
                ui_log("📰 — Analyse macro-économique : actualités financières + FinBERT + Claude Haiku…")
                ui_log("💬 — Analyse du sentiment des investisseurs : ChromaDB + Claude Haiku…")
                ui_log("📊 — Analyse technique et calcul du risque : indicateurs (RSI, MACD…) + Ollama…")
                ui_log("---")
                ui_log("_⏳ Les agents 2, 3 et 4 travaillent en parallèle. "
                       "Consultez le terminal pour le détail en temps réel._")

                # Appel bloquant — les logs détaillés (scores par ETF, timings)
                # s'affichent dans le terminal grâce au logging.basicConfig force=True
                final_state = graph.invoke(initial_state)

                ui_log("")
                ui_log("🧠 — Optimisation du portefeuille et génération de la recommandation (Claude Sonnet)…")
                ui_log("")
                ui_log("✅ **Analyse terminée !** Les résultats sont prêts.")

                st.session_state["results"] = final_state
                status.update(label="✅ Analyse terminée avec succès !", state="complete")
                st.rerun()

            except Exception as e:
                import traceback
                detail = traceback.format_exc()
                ui_log(f"\n❌ **Erreur** : `{e}`")
                print(detail, flush=True)   # traceback complet dans le terminal
                status.update(label="❌ Erreur pendant l'analyse", state="error")
                st.error(
                    f"❌ Une erreur s'est produite : **{e}**\n\n"
                    "Consultez le terminal pour le traceback complet.\n\n"
                    "Points à vérifier :\n"
                    "- Clés API dans `.env` (ANTHROPIC_API_KEY, FINNHUB_API_KEY, NEWSAPI_KEY)\n"
                    "- PyTorch installé (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)\n"
                    "- Ollama lancé (`ollama serve` + `ollama pull mistral`)"
                )

    if st.button("← Modifier mon profil"):
        st.session_state.pop("kyc_done", None)
        st.rerun()


else:
    # ── QUESTIONNAIRE KYC ─────────────────────────────────────────────────────
    st.subheader("Votre profil investisseur — 10 questions")
    st.caption("Ces questions permettent de personnaliser l'allocation selon votre situation.")

    with st.form("kyc_form"):

        # ── Bloc 1 : Aversion au risque ──────────────────────────────────────
        st.markdown("### Bloc 1 — Aversion au risque")

        q1 = st.radio(
            "**Q1** — Vous disposez de 1 000 € à placer sur 8 ans. Quel placement choisissez-vous ?",
            options=["A", "B", "C"],
            format_func=lambda x: {
                "A": "A — Risque faible : gain modeste, capital protégé",
                "B": "B — Risque moyen : gains potentiels plus élevés, pertes limitées",
                "C": "C — Risque élevé : gains très élevés possibles, pertes significatives possibles",
            }[x],
            horizontal=False,
        )

        q2 = st.radio(
            "**Q2** — Votre portefeuille perd 20% en un mois. Que faites-vous ?",
            options=["Vends tout", "J'attends", "Je renforce"],
            format_func=lambda x: {
                "Vends tout":  "Je vends tout",
                "J'attends":   "J'attends, je fais confiance au long terme",
                "Je renforce": "Je renforce mes positions, c'est une opportunité",
            }[x],
        )

        q3 = st.radio(
            "**Q3** — 50% de chances de doubler votre capital, 50% de le réduire d'un tiers. Acceptez-vous ?",
            options=["Non, je refuse", "Oui, j'accepte"],
        )

        q4 = st.radio(
            "**Q4** — Même logique mais 50% de chances de perdre la moitié. Acceptez-vous toujours ?",
            options=["Non, je refuse", "Oui, j'accepte"],
        )

        st.divider()

        # ── Bloc 2 : Capacité financière ─────────────────────────────────────
        st.markdown("### Bloc 2 — Capacité financière")

        q5 = st.radio(
            "**Q5** — Si votre situation financière se dégradait, quelle facilité auriez-vous à vous adapter ?",
            options=["Très peu", "Peu", "Assez grande"],
            format_func=lambda x: {
                "Très peu":     "Très peu de facilité",
                "Peu":          "Peu de facilité",
                "Assez grande": "Assez grande facilité",
            }[x],
        )

        st.divider()

        # ── Bloc 3 : Horizon d'investissement ────────────────────────────────
        st.markdown("### Bloc 3 — Horizon d'investissement")

        q6 = st.radio(
            "**Q6** — Sur combien de temps souhaitez-vous investir ?",
            options=["Moins de 2 ans", "2 à 5 ans", "5 à 10 ans", "Plus de 10 ans"],
        )

        q7 = st.radio(
            "**Q7** — Aurez-vous besoin de récupérer cet argent dans les 2 prochaines années ?",
            options=["Oui, probablement", "Non"],
        )

        st.divider()

        # ── Bloc 4 : Connaissance des actifs ─────────────────────────────────
        st.markdown("### Bloc 4 — Connaissance des actifs")
        st.caption(
            "Cochez les types de placements que vous connaissez suffisamment "
            "pour accepter d'y investir. Pas besoin de connaissances techniques — "
            "répondez selon votre expérience personnelle."
        )

        q8_labels = st.multiselect(
            "**Q8** — Parmi les placements suivants, lesquels vous semblent familiers ?",
            options=Q8_LABELS,
            help="Sélection multiple. Les actifs non sélectionnés peuvent quand même "
                 "apparaître dans votre portefeuille si l'algorithme les juge pertinents.",
        )
        # Conversion immédiate en clés internes pour portfolio.py (_filter_eligible_assets)
        q8 = [Q8_LABEL_TO_KEY[label] for label in q8_labels]

        st.divider()

        # ── Bloc 5 : ESG et objectif ──────────────────────────────────────────
        st.markdown("### Bloc 5 — Préférences ESG et objectif")

        q9 = st.radio(
            "**Q9** — Quelle importance accordez-vous aux critères ESG (environnement, social, gouvernance) ?",
            options=["Aucune", "Modérée", "Forte"],
            format_func=lambda x: {
                "Aucune":  "Aucune — seule la performance compte",
                "Modérée": "Modérée — je préfère les actifs ESG à performance équivalente",
                "Forte":   "Forte — j'exclus les actifs non-ESG même si la performance est moindre",
            }[x],
        )

        q10 = st.radio(
            "**Q10** — Quel est votre objectif principal ?",
            options=["Protéger mon capital", "Équilibrer rendement et risque",
                     "Maximiser le rendement"],
        )

        st.divider()
        submitted = st.form_submit_button(
            "Valider mon profil →", type="primary", use_container_width=True
        )

    # ── Traitement à la soumission ────────────────────────────────────────────
    if submitted:
        answers = {
            "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "q5": q5, "q6": q6, "q7": q7,
            "q8": q8,       # déjà converti en clés internes (ex. ["Nasdaq-100", "Gold"])
            "q9": q9, "q10": q10,
        }
        score, profil, risk_score, horizon = compute_kyc_score(answers)

        st.session_state["kyc_score"]   = score
        st.session_state["kyc_profile"] = {
            "profil":       profil,
            "risk_score":   risk_score,
            "horizon":      horizon,
            "goal":         q10,
            "esg":          q9,
            "known_assets": q8,   # clés internes directement utilisables par portfolio.py
        }
        st.session_state["kyc_done"] = True
        st.rerun()