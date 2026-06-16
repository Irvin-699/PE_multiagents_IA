"""
app.py — Interface Streamlit FinAgent P26
Lancement : python -m streamlit run app.py
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Votre Profil Investisseur",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 55%, #7c3aed 100%);
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 32px;
    color: white;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.28);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.4px; }
.hero p  { font-size: 1.05rem; opacity: 0.82; margin: 10px 0 0 0; font-weight: 400; }

/* ── En-tête de section ── */
.section-card {
    border-radius: 14px;
    padding: 16px 22px 14px 22px;
    margin: 28px 0 18px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-card .s-icon { font-size: 1.5rem; }
.section-card h3 { margin: 0; font-size: 1rem; font-weight: 700; }
.section-card p  { margin: 2px 0 0 0; font-size: 0.82rem; opacity: 0.75; }

.s-pref    { background: #ecfdf5; border-left: 5px solid #10b981; color: #065f46; }
.s-risque  { background: #fffbeb; border-left: 5px solid #f59e0b; color: #78350f; }
.s-capa    { background: #eff6ff; border-left: 5px solid #3b82f6; color: #1e3a8a; }
.s-horizon { background: #faf5ff; border-left: 5px solid #8b5cf6; color: #4c1d95; }
.s-actifs  { background: #ecfeff; border-left: 5px solid #06b6d4; color: #164e63; }
.s-esg     { background: #f0fdf4; border-left: 5px solid #22c55e; color: #14532d; }

/* ── Barre de progression ── */
.prog-wrap {
    background: #e5e7eb;
    border-radius: 99px;
    height: 6px;
    margin-bottom: 28px;
    overflow: hidden;
}
.prog-fill {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    height: 6px;
    border-radius: 99px;
}

/* ── Étapes ── */
.steps {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.step {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #eef2ff;
    color: #4f46e5;
    border: 1px solid #c7d2fe;
}

/* ── Badge profil ── */
.badge { display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 1rem; }
.badge-prudent   { background: #d1fae5; color: #065f46; }
.badge-equilibre { background: #fef3c7; color: #92400e; }
.badge-dynamique { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def hero(subtitle: str = "Quelques minutes pour construire votre stratégie personnalisée"):
    st.markdown(f"""
    <div class="hero">
        <h1>💼 Votre Profil Investisseur</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section(cls: str, icone: str, titre: str, desc: str = ""):
    st.markdown(f"""
    <div class="section-card {cls}">
        <span class="s-icon">{icone}</span>
        <div><h3>{titre}</h3>{"<p>" + desc + "</p>" if desc else ""}</div>
    </div>
    """, unsafe_allow_html=True)


def compute_kyc_score(answers: dict) -> tuple[int, str, int, str]:
    score = 0
    score += {"A": 1, "B": 2, "C": 3}[answers["q1"]]
    score += {"Vends tout": 1, "J'attends": 2, "Je renforce": 3}[answers["q2"]]
    score += 1 if answers["q3"] == "Non, je refuse" else 3
    score += 1 if answers["q4"] == "Non, je refuse" else 3
    score += {"Très peu": 1, "Peu": 2, "Assez grande": 3}[answers["q5"]]
    score += {"Moins de 2 ans": 1, "2 à 5 ans": 2, "5 à 10 ans": 3, "Plus de 10 ans": 4}[answers["q6"]]
    score += 1 if answers["q7"] == "Oui, probablement" else 3
    n = len(answers["q8"])
    score += 1 if n <= 2 else (2 if n <= 6 else 3)
    score += {"Aucune": 1, "Modérée": 2, "Forte": 3}[answers["q9"]]
    score += {"Protéger mon capital": 1, "Équilibrer rendement et risque": 2, "Maximiser le rendement": 3}[answers["q10"]]

    if score <= 16:
        profil = "Prudent"
        risk_score = 1 + round((score - 10) / (16 - 10) * 2)
    elif score <= 23:
        profil = "Équilibré"
        risk_score = 4 + round((score - 17) / (23 - 17) * 2)
    else:
        profil = "Dynamique"
        risk_score = 7 + round((score - 24) / (31 - 24) * 3)

    horizon_map = {"Moins de 2 ans": "short", "2 à 5 ans": "medium",
                   "5 à 10 ans": "long", "Plus de 10 ans": "vlong"}
    horizon = horizon_map[answers["q6"]]
    if answers["q7"] == "Oui, probablement":
        horizon = "short"

    return score, profil, max(1, min(10, risk_score)), horizon


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING : 3 états de l'app
# ─────────────────────────────────────────────────────────────────────────────

if "results" in st.session_state:

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════
    profile = st.session_state["kyc_profile"]
    results = st.session_state["results"]
    score   = st.session_state["kyc_score"]

    hero("Voici l'analyse complète de votre portefeuille")

    icone_profil = {"Prudent": "🟢", "Équilibré": "🟡", "Dynamique": "🔴"}[profile["profil"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profil", f"{icone_profil} {profile['profil']}")
    c2.metric("Score KYC", f"{score} / 31")
    c3.metric("Tolérance au risque", f"{profile['risk_score']} / 10")
    c4.metric("Horizon", profile["horizon"])

    st.divider()

    tab1, tab2, tab3 = st.tabs(["💬 Recommandation", "📊 Allocation & Sentiment", "📈 Risque Technique"])

    with tab1:
        reco = results.get("final_recommendation", "")
        if reco:
            st.markdown(reco)
        else:
            st.warning("Aucune recommandation — vérifiez votre clé ANTHROPIC_API_KEY.")

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
            if sent:
                df_s = (pd.DataFrame({"ETF": list(sent.keys()),
                                      "Sentiment": list(sent.values())})
                          .sort_values("Sentiment", ascending=False))
                st.bar_chart(df_s.set_index("ETF"), use_container_width=True)
                st.dataframe(df_s.style.format({"Sentiment": "{:+.3f}"}),
                             hide_index=True, use_container_width=True)
            else:
                st.warning("Scores sentiment non disponibles.")

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

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — RÉSUMÉ PROFIL + LANCEMENT
    # ══════════════════════════════════════════════════════════════════════════
    profile = st.session_state["kyc_profile"]
    score   = st.session_state["kyc_score"]

    hero("Votre profil est prêt — passons à l'analyse")

    icone = {"Prudent": "🟢", "Équilibré": "🟡", "Dynamique": "🔴"}[profile["profil"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profil", f"{icone} {profile['profil']}")
    c2.metric("Score KYC", f"{score} / 31")
    c3.metric("Tolérance au risque", f"{profile['risk_score']} / 10")
    c4.metric("Horizon", profile["horizon"])

    st.divider()

    col_g, col_s = st.columns(2)
    with col_g:
        zones = profile.get("geo_zones", [])
        st.markdown("**🌍 Préférences géographiques**")
        if zones:
            for z in zones:
                st.markdown(f"- {z}")
        else:
            st.markdown("_Aucune préférence particulière_")

    with col_s:
        secteurs = profile.get("sectors", [])
        st.markdown("**🏭 Secteurs de prédilection**")
        if secteurs:
            for s in secteurs:
                st.markdown(f"- {s}")
        else:
            st.markdown("_Non renseigné_")

    st.divider()

    st.markdown(
        "Les **5 agents IA** vont maintenant analyser votre portefeuille. "
        "⏱ Cette étape prend **1 à 3 minutes**."
    )

    if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours — Agents 1 → 2 → 3 → 4 → 5 …"):
            try:
                from main import build_graph
                from core.config import TICKERS

                graph         = build_graph()
                initial_state = {
                    "assets":       list(TICKERS.values()),
                    "user_profile": {
                        "risk_score": profile["risk_score"],
                        "horizon":    profile["horizon"],
                        "goal":       profile["goal"],
                        "esg":        profile.get("esg", "Modérée"),
                        "geo_zones":  profile.get("geo_zones", []),
                        "sectors":    profile.get("sectors", []),
                    },
                    "market_data":          {},
                    "macro_context":        {},
                    "sentiment_score":      {},
                    "technical_risk":       {},
                    "portfolio_allocation": {},
                    "final_recommendation": "",
                }
                final_state = graph.invoke(initial_state)
                st.session_state["results"] = final_state
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    if st.button("← Modifier mes réponses"):
        st.session_state.pop("kyc_done", None)
        st.rerun()


else:

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — QUESTIONNAIRE KYC
    # ══════════════════════════════════════════════════════════════════════════
    hero("Quelques minutes pour construire votre stratégie personnalisée")

    st.markdown("""
    <div class="prog-wrap"><div class="prog-fill" style="width: 15%"></div></div>
    <div class="steps">
        <span class="step">1 · Vos préférences</span>
        <span class="step">2 · Tolérance au risque</span>
        <span class="step">3 · Capacité financière</span>
        <span class="step">4 · Horizon</span>
        <span class="step">5 · Actifs connus</span>
        <span class="step">6 · ESG & Objectif</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("kyc_form"):

        # ── Bloc 1 : Préférences personnelles ────────────────────────────────
        section("s-pref", "🌍", "Vos préférences", "Zones géographiques et secteurs qui vous parlent")

        q_geo_pref = st.radio(
            "**Q1 —** Avez-vous des préférences géographiques pour investir ?",
            options=["Non, pas de préférence particulière", "Oui, j'ai des préférences"],
        )

        q_geo_zones = st.multiselect(
            "**Q1b —** Si oui, quelles zones géographiques vous intéressent le plus ?",
            options=[
                "Amérique du Nord",
                "Europe",
                "Asie-Pacifique",
                "Marchés émergents (Asie, Afrique, Amérique latine)",
                "Monde entier (exposition globale)",
            ],
            help="Laissez vide si vous n'avez pas de préférence.",
        )

        q_sectors = st.multiselect(
            "**Q2 —** Dans quels secteurs avez-vous le plus de connaissances ou d'intérêt ?",
            options=[
                "Technologie & IA",
                "Finance & Banque",
                "Énergie propre & Environnement",
                "Immobilier",
                "Obligations & Taux d'intérêt",
                "Matières premières & Or",
                "Crypto-actifs",
                "Marchés émergents",
            ],
            help="Sélection multiple — influence la personnalisation de l'analyse.",
        )

        # ── Bloc 2 : Aversion au risque ──────────────────────────────────────
        section("s-risque", "🎯", "Tolérance au risque", "Comment réagissez-vous face aux fluctuations ?")

        q1 = st.radio(
            "**Q3 —** Vous disposez de 1 000 € à placer sur 8 ans. Quel placement choisissez-vous ?",
            options=["A", "B", "C"],
            format_func=lambda x: {
                "A": "A · Risque faible — gain modeste, capital protégé",
                "B": "B · Risque moyen — gains potentiels plus élevés, pertes limitées",
                "C": "C · Risque élevé — gains très élevés possibles, pertes significatives possibles",
            }[x],
        )

        q2 = st.radio(
            "**Q4 —** Votre portefeuille perd 20 % en un mois. Que faites-vous ?",
            options=["Vends tout", "J'attends", "Je renforce"],
            format_func=lambda x: {
                "Vends tout":  "Je vends tout immédiatement",
                "J'attends":   "J'attends — je fais confiance au long terme",
                "Je renforce": "Je renforce mes positions — c'est une opportunité",
            }[x],
        )

        q3 = st.radio(
            "**Q5 —** 50 % de chances de doubler votre capital, 50 % de le réduire d'un tiers. Acceptez-vous ?",
            options=["Non, je refuse", "Oui, j'accepte"],
        )

        q4 = st.radio(
            "**Q6 —** Même logique, mais 50 % de chances de perdre la moitié. Acceptez-vous toujours ?",
            options=["Non, je refuse", "Oui, j'accepte"],
        )

        # ── Bloc 3 : Capacité financière ─────────────────────────────────────
        section("s-capa", "💰", "Capacité financière", "Votre marge de manœuvre en cas de coup dur")

        q5 = st.radio(
            "**Q7 —** Si votre situation financière se dégradait, quelle facilité auriez-vous à vous adapter ?",
            options=["Très peu", "Peu", "Assez grande"],
            format_func=lambda x: {
                "Très peu":     "Très peu de facilité — situation financière fragile",
                "Peu":          "Peu de facilité — marge de manœuvre limitée",
                "Assez grande": "Assez grande facilité — revenus stables et épargne disponible",
            }[x],
        )

        # ── Bloc 4 : Horizon d'investissement ────────────────────────────────
        section("s-horizon", "⏳", "Horizon d'investissement", "Sur quelle durée envisagez-vous de placer ?")

        q6 = st.radio(
            "**Q8 —** Sur combien de temps souhaitez-vous investir ?",
            options=["Moins de 2 ans", "2 à 5 ans", "5 à 10 ans", "Plus de 10 ans"],
        )

        q7 = st.radio(
            "**Q9 —** Aurez-vous besoin de récupérer cet argent dans les 2 prochaines années ?",
            options=["Oui, probablement", "Non"],
        )

        # ── Bloc 5 : Connaissance des actifs ─────────────────────────────────
        section("s-actifs", "🧠", "Connaissance des actifs", "Quels instruments financiers maîtrisez-vous ?")

        q8 = st.multiselect(
            "**Q10 —** Parmi nos 10 actifs, lesquels connaissez-vous suffisamment pour y investir ?",
            options=[
                "Nasdaq-100 (EQQQ.L)",
                "MSCI Emerging Markets (EIMI.L)",
                "Finance Europe (EXH2.DE)",
                "World SRI / ESG (SUWS.L)",
                "Obligations US 10+ ans (LUTR.L)",
                "Bund 1-3 ans (DBXP.DE)",
                "Énergies propres (ICLN)",
                "REITs immobilier (EPRA.PA)",
                "Or (GOLD.PA)",
                "Bitcoin (BTC-USD)",
            ],
            help="Sélectionnez tous les actifs que vous connaissez bien.",
        )

        # ── Bloc 6 : ESG et objectif ──────────────────────────────────────────
        section("s-esg", "🌱", "ESG & Objectif", "Vos valeurs et votre ambition d'investissement")

        q9 = st.radio(
            "**Q11 —** Quelle importance accordez-vous aux critères ESG ?",
            options=["Aucune", "Modérée", "Forte"],
            format_func=lambda x: {
                "Aucune":  "Aucune — seule la performance compte",
                "Modérée": "Modérée — je préfère les actifs ESG à performance équivalente",
                "Forte":   "Forte — j'exclus les actifs non-ESG même si la performance est moindre",
            }[x],
        )

        q10 = st.radio(
            "**Q12 —** Quel est votre objectif principal ?",
            options=["Protéger mon capital", "Équilibrer rendement et risque",
                     "Maximiser le rendement"],
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "✅ Valider mon profil", type="primary", use_container_width=True
        )

    # ── Traitement à la soumission ────────────────────────────────────────────
    if submitted:
        answers = {
            "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "q5": q5, "q6": q6, "q7": q7, "q8": q8,
            "q9": q9, "q10": q10,
        }
        score, profil, risk_score, horizon = compute_kyc_score(answers)

        geo_zones = q_geo_zones if q_geo_pref == "Oui, j'ai des préférences" else []

        st.session_state["kyc_score"]   = score
        st.session_state["kyc_profile"] = {
            "profil":       profil,
            "risk_score":   risk_score,
            "horizon":      horizon,
            "goal":         q10,
            "esg":          q9,
            "geo_zones":    geo_zones,
            "sectors":      q_sectors,
            "known_assets": q8,
        }
        st.session_state["kyc_done"] = True
        st.rerun()
