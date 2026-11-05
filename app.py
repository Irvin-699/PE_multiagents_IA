"""
app.py — Interface Streamlit de Wisevest
Onboarding investisseur (5 étapes) + pipeline multi-agents + tableau de bord.

Lancement : python -m streamlit run app.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

# ── Logging : force les messages agents à apparaître dans le terminal ─────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    force  = True,
    stream = sys.stdout,
)

# ── Configuration page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wisevest — Gestion de portefeuille",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — palette sobre finance (bleu nuit / ardoise, un accent)
# Séries de graphiques : palette catégorielle validée (validate_palette.js PASS)
# ─────────────────────────────────────────────────────────────────────────────

NAVY      = "#16263f"   # bleu nuit — header, titres
SLATE     = "#52514e"   # encre secondaire
MUTED     = "#898781"   # axes, légendes
ACCENT    = "#1c5cab"   # accent unique (boutons, liens, barres)
SURFACE   = "#fcfcfb"   # surface des graphiques
HAIRLINE  = "#e1e0d9"   # grilles, séparateurs
POSITIVE  = "#2a78d6"   # pôle divergent positif
NEGATIVE  = "#e34948"   # pôle divergent négatif

# Palette catégorielle (8 emplacements, ordre fixe — jamais cyclée)
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
OTHER_GRAY  = "#c3c2b7"   # tranche « Autres » au-delà de 8 positions

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'

st.markdown(f"""
<style>
/* ── Base ── */
#MainMenu, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.2rem; max-width: 880px; }}

/* ── Header produit ── */
.wv-header {{
    background: {NAVY};
    border-radius: 10px;
    padding: 26px 32px;
    margin-bottom: 28px;
}}
/* Variante héro (page questionnaire) : photo de marché sous overlay bleu nuit.
   La couleur {NAVY} reste le fond de secours si l'image ne charge pas. */
.wv-header.hero {{
    background:
        linear-gradient(rgba(13, 23, 41, 0.78), rgba(13, 23, 41, 0.86)),
        {NAVY}
        url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1600&q=80&auto=format&fit=crop')
        center / cover no-repeat;
    padding: 44px 36px;
}}
.wv-header .wv-brand {{
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.4px;
}}
.wv-header.hero .wv-brand {{ font-size: 1.9rem; }}
.wv-header .wv-brand span {{ color: #86b6ef; }}
.wv-header .wv-tagline {{
    color: rgba(255,255,255,0.72);
    font-size: 0.92rem;
    margin-top: 4px;
}}
.wv-header.hero .wv-tagline {{ font-size: 1rem; max-width: 520px; }}

/* ── Bandeau image discret (écran d'attente) ── */
.wv-banner {{
    width: 100%;
    height: 130px;
    border-radius: 10px;
    margin-bottom: 20px;
    background:
        linear-gradient(rgba(13, 23, 41, 0.35), rgba(13, 23, 41, 0.45)),
        url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1600&q=80&auto=format&fit=crop')
        center / cover no-repeat;
}}

/* ── Cartes de métriques ── */
.wv-metrics {{ display: flex; gap: 14px; margin: 6px 0 10px 0; }}
.wv-card {{
    flex: 1;
    background: #ffffff;
    border: 1px solid {HAIRLINE};
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(22,38,63,0.06);
}}
.wv-card .wv-k-label {{
    color: {MUTED};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.wv-card .wv-k-value {{
    color: {NAVY};
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 2px;
}}
.wv-card .wv-k-sub {{ color: {SLATE}; font-size: 0.8rem; margin-top: 2px; }}

/* ── Bandeau profil ── */
.wv-profile {{
    background: #ffffff;
    border: 1px solid {HAIRLINE};
    border-left: 4px solid {ACCENT};
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 20px;
    color: {SLATE};
    font-size: 0.95rem;
}}
.wv-profile b {{ color: {NAVY}; }}

/* ── Étapes du pipeline ── */
.wv-steps {{ margin: 10px 0 6px 0; }}
.wv-step {{
    display: flex; align-items: center; gap: 12px;
    padding: 9px 4px; color: {SLATE}; font-size: 0.95rem;
}}
.wv-step .dot {{
    width: 22px; height: 22px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; flex: none;
    border: 2px solid {HAIRLINE}; color: {MUTED}; background: #fff;
}}
.wv-step.done .dot    {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.wv-step.current .dot {{ border-color: {ACCENT}; color: {ACCENT}; }}
.wv-step.current      {{ color: {NAVY}; font-weight: 600; }}
.wv-step.done         {{ color: {MUTED}; }}

/* ── Typo questionnaire ── */
.wv-section-title {{
    color: {NAVY}; font-size: 1.25rem; font-weight: 700;
    margin: 6px 0 2px 0;
}}
.wv-section-sub {{ color: {SLATE}; font-size: 0.92rem; margin-bottom: 14px; }}
div[role="radiogroup"] label {{ padding: 4px 0; }}

/* ── Footer ── */
.wv-footer {{
    border-top: 1px solid {HAIRLINE};
    margin-top: 48px; padding: 18px 4px 8px 4px;
    color: {MUTED}; font-size: 0.8rem;
}}
</style>
""", unsafe_allow_html=True)


def render_header(hero: bool = False) -> None:
    """Header produit. `hero=True` (questionnaire) : photo de marché en fond ;
    sinon bandeau bleu nuit sobre — les résultats laissent la place aux graphiques."""
    css_class = "wv-header hero" if hero else "wv-header"
    st.markdown(
        f'<div class="{css_class}">'
        '<div class="wv-brand">Wise<span>vest</span></div>'
        '<div class="wv-tagline">Construisez un portefeuille aligné sur votre profil, '
        'votre horizon et les conditions de marché.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="wv-footer">'
        'Wisevest est un outil d\'aide à la décision et ne constitue pas un conseil '
        'en investissement au sens de la réglementation. Les performances passées ne '
        'préjugent pas des performances futures ; tout investissement comporte un '
        'risque de perte en capital.<br>'
        '© 2026 Wisevest · Projet académique · Photographies : Unsplash'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES PLOTLY — thème commun sobre
# ─────────────────────────────────────────────────────────────────────────────

def _base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, color=SLATE, size=13),
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=HAIRLINE,
                        font=dict(family=FONT_STACK, color=NAVY, size=13)),
    )
    return fig


def donut_allocation(alloc: dict[str, float]) -> go.Figure:
    """Donut d'allocation — top 8 positions + « Autres », couleurs à ordre fixe."""
    items = sorted(
        ((k, v) for k, v in alloc.items() if v > 0.05),
        key=lambda x: x[1], reverse=True,
    )
    top, rest = items[:8], items[8:]
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    colors = CATEGORICAL[:len(top)]
    if rest:
        labels.append("Autres")
        values.append(round(sum(v for _, v in rest), 2))
        colors.append(OTHER_GRAY)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        sort=False,
        direction="clockwise",
        marker=dict(colors=colors, line=dict(color=SURFACE, width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="#ffffff"),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{value:.1f} % du portefeuille<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(items)}</b><br><span style='font-size:12px'>positions</span>",
        showarrow=False, font=dict(family=FONT_STACK, size=22, color=NAVY),
    )
    fig.update_layout(
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=12, color=SLATE)),
    )
    return _base_layout(fig, height=340)


def hbar_magnitude(data: dict[str, float], unit: str = "%",
                   fmt: str = ".1f") -> go.Figure:
    """Barres horizontales triées — magnitude : une seule teinte (accent)."""
    items  = sorted(data.items(), key=lambda x: x[1])
    names  = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=POSITIVE, cornerradius=4),
        text=[f"{v:{fmt}} {unit}".strip() for v in values],
        textposition="outside",
        textfont=dict(size=11, color=MUTED),
        hovertemplate="<b>%{y}</b><br>%{x:" + fmt + "} " + unit + "<extra></extra>",
    ))
    fig.update_xaxes(showgrid=True, gridcolor=HAIRLINE, zeroline=False,
                     showticklabels=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=NAVY))
    fig.update_layout(bargap=0.45, showlegend=False)
    return _base_layout(fig, height=max(300, 32 * len(items)))


def hbar_diverging(data: dict[str, float]) -> go.Figure:
    """Barres horizontales divergentes — polarité : bleu (+) / rouge (−) / gris (≈0)."""
    items  = sorted(data.items(), key=lambda x: x[1])
    names  = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [POSITIVE if v > 0.05 else NEGATIVE if v < -0.05 else OTHER_GRAY
              for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=MUTED),
        hovertemplate="<b>%{y}</b><br>Score : %{x:+.2f}<extra></extra>",
    ))
    fig.update_xaxes(range=[-1.15, 1.15], showgrid=True, gridcolor=HAIRLINE,
                     zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                     tickvals=[-1, -0.5, 0, 0.5, 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=NAVY))
    fig.update_layout(bargap=0.45, showlegend=False)
    return _base_layout(fig, height=max(300, 32 * len(items)))


def metric_cards(cards: list[tuple[str, str, str]]) -> None:
    """Rangée de cartes de métriques : (label, valeur, sous-texte)."""
    html = '<div class="wv-metrics">'
    for label, value, sub in cards:
        html += (f'<div class="wv-card"><div class="wv-k-label">{label}</div>'
                 f'<div class="wv-k-value">{value}</div>'
                 f'<div class="wv-k-sub">{sub}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KYC — questions, scoring, mapping actifs
# ─────────────────────────────────────────────────────────────────────────────

Q8_LABEL_TO_KEY = {
    "Actions technologiques mondiales (Apple, Microsoft, Google…)":  "Nasdaq-100",
    "Actions des pays émergents (Inde, Brésil, Chine…)":             "MSCI EM IMI",
    "Actions du secteur bancaire et financier européen":              "STOXX Eur600 Finance",
    "Actions du secteur santé et pharmaceutique mondial":             "Global Healthcare",
    "Obligations d'État américaines à long terme":                    "US Treasury 10+Y",
    "Obligations d'État allemandes à court terme":                    "Bund 1-3Y",
    "Actions des énergies renouvelables":                             "Clean Energy",
    "Immobilier coté en bourse (bureaux, commerces…)":                "Global REITs",
    "Or (valeur refuge)":                                             "Gold",
    "Bitcoin":                                                        "Bitcoin",
}
Q8_LABELS = list(Q8_LABEL_TO_KEY.keys())


def compute_kyc_score(answers: dict) -> tuple[int, str, int, str]:
    """
    Calcule le score total (10-31), le profil et le risk_score (1-10).
    q8 doit déjà contenir des clés internes (ex. "Nasdaq-100"), pas des libellés.
    Retourne : (score, profil_label, risk_score, horizon_code)
    """
    score = 0
    score += {"A": 1, "B": 2, "C": 3}[answers["q1"]]
    score += {"Vends tout": 1, "J'attends": 2, "Je renforce": 3}[answers["q2"]]
    score += 1 if answers["q3"] == "Non, je refuse" else 3
    score += 1 if answers["q4"] == "Non, je refuse" else 3
    score += {"Très peu": 1, "Peu": 2, "Assez grande": 3}[answers["q5"]]
    score += {"Moins de 2 ans": 1, "2 à 5 ans": 2, "5 à 10 ans": 3,
              "Plus de 10 ans": 4}[answers["q6"]]
    score += 1 if answers["q7"] == "Oui, probablement" else 3
    n = len(answers["q8"])
    score += 1 if n <= 2 else (2 if n <= 6 else 3)
    score += {"Aucune": 1, "Modérée": 2, "Forte": 3}[answers["q9"]]
    score += {"Protéger mon capital": 1, "Équilibrer rendement et risque": 2,
              "Maximiser le rendement": 3}[answers["q10"]]

    if score <= 16:
        profil     = "Prudent"
        risk_score = 1 + round((score - 10) / (16 - 10) * 2)   # 1-3
    elif score <= 23:
        profil     = "Équilibré"
        risk_score = 4 + round((score - 17) / (23 - 17) * 2)   # 4-6
    else:
        profil     = "Dynamique"
        risk_score = 7 + round((score - 24) / (31 - 24) * 3)   # 7-10

    horizon_map = {"Moins de 2 ans": "short", "2 à 5 ans": "medium",
                   "5 à 10 ans": "long", "Plus de 10 ans": "vlong"}
    horizon = horizon_map[answers["q6"]]
    if answers["q7"] == "Oui, probablement":   # besoin de liquidité → court terme
        horizon = "short"

    return score, profil, max(1, min(10, risk_score)), horizon


# ── Assistant en 5 étapes ─────────────────────────────────────────────────────

KYC_STEPS = [
    ("Votre rapport au risque",
     "Il n'y a pas de bonne ou de mauvaise réponse — répondez spontanément."),
    ("Votre capacité financière",
     "Votre situation détermine le niveau de risque soutenable."),
    ("Votre horizon de placement",
     "Le temps est votre principal allié — ou votre principale contrainte."),
    ("Les placements que vous connaissez",
     "Nous ne proposons que des actifs que vous comprenez."),
    ("Vos préférences et votre objectif",
     "Dernière étape avant votre analyse personnalisée."),
]


def _radio(label: str, options: list[str], key: str,
           format_func=None) -> str | None:
    """st.radio pré-rempli depuis les réponses déjà saisies, sinon vide."""
    answers = st.session_state.setdefault("kyc_answers", {})
    stored  = answers.get(key)
    index   = options.index(stored) if stored in options else None
    kwargs  = {"format_func": format_func} if format_func else {}
    return st.radio(label, options, index=index, key=f"w_{key}", **kwargs)


def render_kyc_wizard() -> None:
    step = st.session_state.setdefault("kyc_step", 0)
    answers = st.session_state.setdefault("kyc_answers", {})

    title, subtitle = KYC_STEPS[step]
    st.progress((step) / len(KYC_STEPS),
                text=f"Étape {step + 1} sur {len(KYC_STEPS)}")
    st.markdown(f'<div class="wv-section-title">{title}</div>'
                f'<div class="wv-section-sub">{subtitle}</div>',
                unsafe_allow_html=True)

    collected: dict = {}

    if step == 0:
        collected["q1"] = _radio(
            "Vous disposez de 1 000 € à placer sur 8 ans. Que choisissez-vous ?",
            ["A", "B", "C"], "q1",
            format_func=lambda x: {
                "A": "La sécurité : gain modeste, capital protégé",
                "B": "Le compromis : gains plus élevés, pertes limitées",
                "C": "La performance : gains importants possibles, pertes aussi",
            }[x])
        collected["q2"] = _radio(
            "Votre portefeuille perd 20 % en un mois. Que faites-vous ?",
            ["Vends tout", "J'attends", "Je renforce"], "q2",
            format_func=lambda x: {
                "Vends tout":  "Je vends tout pour limiter les dégâts",
                "J'attends":   "J'attends, je fais confiance au long terme",
                "Je renforce": "J'investis davantage, c'est une opportunité",
            }[x])
        collected["q3"] = _radio(
            "On vous propose 50 % de chances de doubler votre capital, "
            "50 % de le réduire d'un tiers. Acceptez-vous ?",
            ["Non, je refuse", "Oui, j'accepte"], "q3")
        collected["q4"] = _radio(
            "Même pari, mais vous pouvez perdre la moitié. Acceptez-vous toujours ?",
            ["Non, je refuse", "Oui, j'accepte"], "q4")

    elif step == 1:
        collected["q5"] = _radio(
            "Si votre situation financière se dégradait, quelle facilité "
            "auriez-vous à vous adapter ?",
            ["Très peu", "Peu", "Assez grande"], "q5",
            format_func=lambda x: {
                "Très peu":     "Très peu — mon budget est déjà serré",
                "Peu":          "Peu — j'ai une petite marge de manœuvre",
                "Assez grande": "Assez grande — j'ai de la réserve",
            }[x])

    elif step == 2:
        collected["q6"] = _radio(
            "Sur combien de temps souhaitez-vous investir ?",
            ["Moins de 2 ans", "2 à 5 ans", "5 à 10 ans", "Plus de 10 ans"], "q6")
        collected["q7"] = _radio(
            "Pensez-vous avoir besoin de cet argent dans les 2 prochaines années ?",
            ["Oui, probablement", "Non"], "q7")

    elif step == 3:
        stored_q8 = answers.get("q8_labels", [])
        collected["q8_labels"] = st.multiselect(
            "Parmi ces placements, lesquels vous sont familiers ?",
            options=Q8_LABELS,
            default=[l for l in stored_q8 if l in Q8_LABELS],
            key="w_q8_labels",
            help="Sélection multiple — répondez selon votre expérience personnelle, "
                 "aucune connaissance technique n'est requise.",
        )

    elif step == 4:
        collected["q9"] = _radio(
            "Quelle importance accordez-vous aux critères environnementaux "
            "et sociaux (ESG) ?",
            ["Aucune", "Modérée", "Forte"], "q9",
            format_func=lambda x: {
                "Aucune":  "Aucune — seule la performance compte",
                "Modérée": "Modérée — à performance égale, je préfère l'ESG",
                "Forte":   "Forte — j'exclus les actifs non responsables",
            }[x])
        collected["q10"] = _radio(
            "Quel est votre objectif principal ?",
            ["Protéger mon capital", "Équilibrer rendement et risque",
             "Maximiser le rendement"], "q10")

    # ── Navigation ────────────────────────────────────────────────────────────
    st.write("")
    col_back, col_next = st.columns([1, 2])

    with col_back:
        if step > 0 and st.button("Retour", use_container_width=True,
                                  key="btn_kyc_back"):
            answers.update({k: v for k, v in collected.items() if v is not None})
            st.session_state["kyc_step"] = step - 1
            st.rerun()

    with col_next:
        is_last  = step == len(KYC_STEPS) - 1
        btn_text = "Voir mon profil" if is_last else "Continuer"
        if st.button(btn_text, type="primary", use_container_width=True,
                     key="btn_kyc_next"):
            missing = [k for k, v in collected.items()
                       if v is None and k != "q8_labels"]
            if missing:
                st.warning("Merci de répondre à toutes les questions avant de continuer.")
            else:
                answers.update(collected)
                if is_last:
                    answers["q8"] = [Q8_LABEL_TO_KEY[l]
                                     for l in answers.get("q8_labels", [])]
                    score, profil, risk_score, horizon = compute_kyc_score(answers)
                    st.session_state["kyc_score"]   = score
                    st.session_state["kyc_profile"] = {
                        "profil":       profil,
                        "risk_score":   risk_score,
                        "horizon":      horizon,
                        "goal":         answers["q10"],
                        "esg":          answers["q9"],
                        "known_assets": answers["q8"],
                    }
                    st.session_state["kyc_done"] = True
                else:
                    st.session_state["kyc_step"] = step + 1
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — exécution avec progression par étape métier
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "Collecte des données de marché",
    "Analyse macro-économique",
    "Analyse du sentiment des investisseurs",
    "Modélisation du risque",
    "Optimisation du portefeuille",
]

AGENT_TIMEOUT = 180   # secondes par étape


def _render_steps(placeholder, done: set[int], current: set[int]) -> None:
    html = '<div class="wv-steps">'
    for i, name in enumerate(PIPELINE_STEPS):
        if i in done:
            cls, dot = "done", "&#10003;"
        elif i in current:
            cls, dot = "current", str(i + 1)
        else:
            cls, dot = "", str(i + 1)
        html += (f'<div class="wv-step {cls}"><span class="dot">{dot}</span>'
                 f'{name}</div>')
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


def execute_pipeline(initial_state: dict, placeholder) -> dict:
    """
    Orchestre les 5 agents avec l'ordre strict des dépendances :
        Agent 1 → Agent 2 → (Agent 3 ∥ Agent 4) → Agent 5

    Chaque agent tourne dans un thread d'executor : les boucles asyncio
    internes des Agents 2 et 3 (asyncio.run) y créent leur propre event
    loop, jamais celle de Streamlit. L'UI est mise à jour entre les étapes
    depuis le thread principal du script.
    """
    from concurrent.futures import ThreadPoolExecutor
    from agents.market_data import market_data_node
    from agents.fundamental import fundamental_node
    from agents.sentiment   import sentiment_node
    from agents.technical   import technical_node
    from agents.portfolio   import portfolio_node

    state = dict(initial_state)
    done: set[int] = set()

    def run_stage(pool, index: int, fn) -> dict:
        try:
            return pool.submit(fn, state).result(timeout=AGENT_TIMEOUT) or {}
        except Exception as e:
            logging.error("[Pipeline] Étape %d (%s) échouée : %s",
                          index + 1, PIPELINE_STEPS[index], e)
            return {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Préchauffage FinBERT en fond : recouvre l'Agent 1 et la collecte
        # réseau de l'Agent 2 (voir main.py). Une fois chargé, le modèle
        # reste en mémoire pour toutes les analyses suivantes de la session.
        from core.models import get_finbert
        pool.submit(get_finbert)

        # Étape 1 — Agent 1 (bloquant : tout le reste en dépend)
        _render_steps(placeholder, done, {0})
        state.update(run_stage(pool, 0, market_data_node))
        if not state.get("market_data"):
            raise RuntimeError("Aucune donnée de marché disponible — "
                               "vérifiez le fichier data/portfolio_daily.parquet.")
        done.add(0)

        # Étape 2 — Agent 2 (écrit ChromaDB, doit finir avant l'Agent 3)
        _render_steps(placeholder, done, {1})
        state.update(run_stage(pool, 1, fundamental_node))
        done.add(1)

        # Étapes 3 ∥ 4 — Agent 3 (lit ChromaDB) et Agent 4 (indépendant)
        _render_steps(placeholder, done, {2, 3})
        f_sent = pool.submit(sentiment_node, state)
        f_tech = pool.submit(technical_node, state)
        for index, future in ((2, f_sent), (3, f_tech)):
            try:
                state.update(future.result(timeout=AGENT_TIMEOUT) or {})
            except Exception as e:
                logging.error("[Pipeline] Étape %d échouée : %s", index + 1, e)
            done.add(index)

        # Étape 5 — Agent 5 (dépend de tout le monde)
        _render_steps(placeholder, done, {4})
        state.update(run_stage(pool, 4, portfolio_node))
        done.add(4)

    _render_steps(placeholder, done, set())
    return state


# ─────────────────────────────────────────────────────────────────────────────
# RÉSULTATS — résumé exécutif + détails en onglets
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_max_drawdown(results: dict, alloc: dict) -> float | None:
    """Max drawdown historique du portefeuille (poids fixes, base quotidienne)."""
    try:
        prices = results.get("market_data", {}).get("prices")
        if not isinstance(prices, pd.DataFrame):
            return None
        cols = [a for a, p in alloc.items() if p > 0 and a in prices.columns]
        if not cols:
            return None
        weights = np.array([alloc[a] for a in cols], dtype=float)
        weights = weights / weights.sum()
        returns = prices[cols].dropna().pct_change().dropna()
        value   = (1 + returns.values @ weights).cumprod()
        peak    = np.maximum.accumulate(value)
        return float(((value - peak) / peak).min())
    except Exception:
        return None


def render_results() -> None:
    profile = st.session_state["kyc_profile"]
    results = st.session_state["results"]
    score   = st.session_state["kyc_score"]

    horizon_labels = {"short": "moins de 2 ans", "medium": "2 à 5 ans",
                      "long": "5 à 10 ans", "vlong": "plus de 10 ans"}

    st.markdown(
        f'<div class="wv-profile">Profil <b>{profile["profil"]}</b> · '
        f'Tolérance au risque <b>{profile["risk_score"]}/10</b> · '
        f'Horizon <b>{horizon_labels.get(profile["horizon"], profile["horizon"])}</b> · '
        f'Score questionnaire {score}/31</div>',
        unsafe_allow_html=True,
    )

    alloc   = results.get("portfolio_allocation", {})
    details = results.get("optimization_details", {})
    metrics = details.get("metrics", {})

    # ── Résumé exécutif : métriques clés + donut ─────────────────────────────
    if metrics:
        mdd = _portfolio_max_drawdown(results, alloc)
        metric_cards([
            ("Rendement attendu", f"{metrics.get('return_annual', 0):.1f} %", "annualisé"),
            ("Volatilité",        f"{metrics.get('volatility_annual', 0):.1f} %", "annualisée"),
            ("Ratio de Sharpe",   f"{metrics.get('sharpe_ratio', 0):.2f}",
             "rendement / risque"),
            ("Perte max. historique",
             f"{mdd * 100:.1f} %" if mdd is not None else "—",
             "depuis 2016"),
        ])

    if alloc:
        st.markdown("#### Allocation proposée")
        st.plotly_chart(donut_allocation(alloc), use_container_width=True,
                        config={"displayModeBar": False},
                        key="chart_allocation_donut")
        df_alloc = (pd.DataFrame({"Actif": list(alloc.keys()),
                                  "Poids (%)": list(alloc.values())})
                    .sort_values("Poids (%)", ascending=False)
                    .reset_index(drop=True))
        st.dataframe(df_alloc.style.format({"Poids (%)": "{:.1f} %"}),
                     hide_index=True, use_container_width=True,
                     key="table_allocation")
        if details.get("method_used"):
            st.caption(f"Méthode d'optimisation : {details['method_used']} · "
                       f"κ = {details.get('kappa', '—')}")
    else:
        st.warning("Allocation non disponible — consultez le terminal.")

    # ── Détails en onglets ────────────────────────────────────────────────────
    tab_reco, tab_market, tab_risk = st.tabs(
        ["Recommandation", "Marchés et sentiment", "Risque"]
    )

    with tab_reco:
        reco = results.get("final_recommendation", "")
        if reco:
            with st.container(border=True):
                st.markdown(reco)
        else:
            st.warning("Recommandation indisponible — vérifiez ANTHROPIC_API_KEY dans .env.")

    with tab_market:
        sent = results.get("sentiment_score", {})
        sent_scores = sent.get("scores", {}) if isinstance(sent, dict) else {}
        macro = results.get("macro_context", {})
        macro_scores = macro.get("scores", {}) if isinstance(macro, dict) else {}

        if macro_scores:
            st.markdown("##### Environnement macro-économique par actif")
            st.caption("Score de −1 (défavorable) à +1 (favorable), "
                       "issu de l'analyse des actualités récentes.")
            st.plotly_chart(hbar_diverging(macro_scores), use_container_width=True,
                            config={"displayModeBar": False},
                            key="chart_macro_bars")
        if sent_scores:
            st.markdown("##### Sentiment des investisseurs par actif")
            st.caption("Score de −1 (pessimisme) à +1 (optimisme), "
                       "mesuré sur le ton des actualités.")
            st.plotly_chart(hbar_diverging(sent_scores), use_container_width=True,
                            config={"displayModeBar": False},
                            key="chart_sentiment_bars")
        if not macro_scores and not sent_scores:
            st.info("Scores macro et sentiment non disponibles.")

    with tab_risk:
        tech = results.get("technical_risk", {})
        vols = tech.get("volatilities", {})
        if vols:
            st.markdown("##### Volatilité annualisée par actif")
            st.plotly_chart(
                hbar_magnitude({k: v * 100 for k, v in vols.items()}),
                use_container_width=True, config={"displayModeBar": False},
                key="chart_volatility_bars")

        sharpe = tech.get("sharpe_ratios", {})
        mdd_by_asset = tech.get("max_drawdowns", {})
        if sharpe:
            st.markdown("##### Indicateurs par actif")
            df_risk = pd.DataFrame({
                "Actif":            list(sharpe.keys()),
                "Sharpe":           [sharpe.get(k) for k in sharpe],
                "Perte max. (%)":   [round((mdd_by_asset.get(k) or 0) * 100, 1)
                                     for k in sharpe],
            }).sort_values("Sharpe", ascending=False).reset_index(drop=True)
            st.dataframe(df_risk.style.format({"Sharpe": "{:.2f}",
                                               "Perte max. (%)": "{:.1f} %"}),
                         hide_index=True, use_container_width=True,
                         key="table_risk_indicators")
        if not tech:
            st.info("Données de risque non disponibles.")

    st.write("")
    if st.button("Refaire le questionnaire", key="btn_restart_quiz"):
        for k in ["results", "kyc_profile", "kyc_score", "kyc_done",
                  "kyc_step", "kyc_answers"]:
            st.session_state.pop(k, None)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ÉCRAN DE LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def render_launch() -> None:
    profile = st.session_state["kyc_profile"]
    score   = st.session_state["kyc_score"]

    st.markdown('<div class="wv-banner"></div>', unsafe_allow_html=True)

    horizon_labels = {"short": "moins de 2 ans", "medium": "2 à 5 ans",
                      "long": "5 à 10 ans", "vlong": "plus de 10 ans"}
    st.markdown(
        f'<div class="wv-profile">Profil <b>{profile["profil"]}</b> · '
        f'Tolérance au risque <b>{profile["risk_score"]}/10</b> · '
        f'Horizon <b>{horizon_labels.get(profile["horizon"], profile["horizon"])}</b> · '
        f'Score questionnaire {score}/31</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Votre analyse personnalisée")
    st.markdown(
        "Cinq moteurs d'analyse vont examiner les marchés, les actualités "
        "économiques, le sentiment des investisseurs et le risque de chaque "
        "actif, puis construire l'allocation adaptée à votre profil. "
        "Comptez environ une minute."
    )

    if st.button("Lancer l'analyse", type="primary", use_container_width=True,
                 key="btn_run_analysis"):
        steps_placeholder = st.empty()
        try:
            from core.config import TICKERS

            initial_state = {
                "assets": list(TICKERS.keys()),
                "user_profile": {
                    "profil":       profile["profil"],
                    "risk_score":   profile["risk_score"],
                    "horizon":      profile["horizon"],
                    "goal":         profile["goal"],
                    "known_assets": profile.get("known_assets",
                                                list(TICKERS.keys())),
                },
                "market_data":          {},
                "macro_context":        {},
                "sentiment_score":      {},
                "technical_risk":       {},
                "portfolio_allocation": {},
                "final_recommendation": "",
            }

            final_state = execute_pipeline(initial_state, steps_placeholder)
            st.session_state["results"] = final_state
            st.rerun()

        except Exception as e:
            import traceback
            print(traceback.format_exc(), flush=True)
            st.error(
                f"Une erreur s'est produite : **{e}**\n\n"
                "Consultez le terminal pour le détail. Points à vérifier : "
                "clés API dans `.env` (voir `.env.example`), dépendances "
                "installées (`pip install -r requirements_PETEM.txt`)."
            )

    if st.button("Modifier mon profil", key="btn_edit_profile"):
        st.session_state.pop("kyc_done", None)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTAGE
# ─────────────────────────────────────────────────────────────────────────────

# Héro avec image uniquement sur le questionnaire ; header sobre ailleurs
# (l'écran de résultats laisse les graphiques porter le visuel).
if "results" in st.session_state:
    render_header(hero=False)
    render_results()
elif "kyc_done" in st.session_state:
    render_header(hero=False)
    render_launch()
else:
    render_header(hero=True)
    render_kyc_wizard()

render_footer()
