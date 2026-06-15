"""
agents/portfolio.py
Agent 5 — Portfolio Manager

Rôle dans le pipeline LangGraph :
    Reçoit les sorties des 4 agents précédents, optimise l'allocation
    du portefeuille selon le profil KYC de l'utilisateur, et génère
    une recommandation en langage naturel via Claude Sonnet.

Pipeline :
    1. Filtrage       → actifs éligibles selon profil KYC
    2. Optimisation   → Minimum Variance (principal) ou Robust Optimization
    3. Ajustement     → pondération tactique par scores macro + sentiment
    4. Contraintes    → bornes min/max selon profil Prudent/Équilibré/Dynamique
    5. Recommandation → Claude Sonnet génère le message final

Méthodes d'optimisation :
    A. Equal Weight         — baseline de référence
    B. Minimum Variance     — minimise σ²(w) sans estimer les rendements futurs
                              (recommandé par le tuteur — le plus robuste)
    C. Robust Optimization  — maximise le rendement ajusté au pire cas
                              (fallback sur MVP si cvxpy échoue)

Note sur le choix MVP :
    Le MVP ne nécessite aucune estimation des rendements futurs — uniquement
    la matrice de covariance. C'est son avantage principal : les estimations
    de rendements futurs sont très bruitées et dégradent l'optimisation.
    En évitant ce biais, MVP produit des allocations plus stables et robustes.

LLM : Claude 3.5 Sonnet (~20€/mois)
"""

import sys
import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from core.state import AgentState

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Contraintes de poids par profil (min%, max% par actif)
PROFILE_CONSTRAINTS = {
    "Prudent": {
        "w_min": 0.02,    # Minimum 2% par actif éligible
        "w_max": 0.20,    # Maximum 20% par actif
        "btc_max": 0.00,  # Bitcoin interdit
        "risky_max": 0.10,# Max 10% par actif risqué (NDX, ICLN, EM)
    },
    "Équilibré": {
        "w_min": 0.02,
        "w_max": 0.25,
        "btc_max": 0.05,
        "risky_max": 0.20,
    },
    "Dynamique": {
        "w_min": 0.02,
        "w_max": 0.35,
        "btc_max": 0.15,
        "risky_max": 0.35,
    },
}

# Actifs considérés comme risqués pour les contraintes de profil
RISKY_ASSETS = {"Nasdaq-100", "MSCI EM IMI", "Clean Energy", "Bitcoin"}

# Pondération de l'ajustement tactique (macro + sentiment) par horizon
TACTICAL_WEIGHT = {
    "short":  0.30,   # Ajustement fort à court terme
    "medium": 0.20,
    "long":   0.10,
    "vlong":  0.05,   # Quasi nul à très long terme
}


# ─── 1. Filtrage des actifs éligibles ────────────────────────────────────────

def _filter_eligible_assets(
    assets       : list[str],
    cov_matrix   : np.ndarray,
    tickers      : list[str],
    user_profile : dict,
) -> tuple[list[str], np.ndarray]:
    """
    Filtre les actifs éligibles selon le profil KYC et retourne
    la sous-matrice de covariance correspondante.

    Règles :
        - Bitcoin exclu pour profil Prudent
        - Actifs non connus de l'utilisateur exclus (q8 KYC)
        - Minimum 3 actifs pour que l'optimisation ait du sens
    """
    profil        = user_profile.get("profil", "Équilibré")
    known_assets  = set(user_profile.get("known_assets", assets))
    eligible_set  = set(user_profile.get("eligible_assets", assets))
    constraints   = PROFILE_CONSTRAINTS.get(profil, PROFILE_CONSTRAINTS["Équilibré"])

    eligible = []
    for asset in assets:
        # Exclure Bitcoin pour profil Prudent
        if asset == "Bitcoin" and constraints["btc_max"] == 0.0:
            logger.info("[Agent 5] Bitcoin exclu — profil %s", profil)
            continue
        # Garder uniquement les actifs connus et éligibles géographiquement/sectoriellement
        if asset not in known_assets and known_assets != set(assets):
            logger.info("[Agent 5] %s exclu — non connu de l'utilisateur", asset)
            continue
        eligible.append(asset)

    # Sécurité : minimum 3 actifs
    if len(eligible) < 3:
        logger.warning("[Agent 5] Moins de 3 actifs éligibles — utilisation de tous les actifs.")
        eligible = assets

    # Extraire la sous-matrice de covariance
    indices  = [tickers.index(a) for a in eligible if a in tickers]
    sub_cov  = cov_matrix[np.ix_(indices, indices)]

    logger.info("[Agent 5] %d actifs éligibles : %s", len(eligible), eligible)
    return eligible, sub_cov


# ─── 2A. Equal Weight (baseline) ─────────────────────────────────────────────

def _equal_weight(assets: list[str]) -> np.ndarray:
    """Allocation equal weight — 1/n par actif."""
    n = len(assets)
    return np.ones(n) / n


# ─── 2B. Minimum Variance Portfolio ──────────────────────────────────────────

def _minimum_variance(
    cov_matrix   : np.ndarray,
    assets       : list[str],
    user_profile : dict,
) -> np.ndarray:
    """
    Minimum Variance Portfolio (MVP) — méthode principale.

    Résout :
        min  w^T Σ w
         w
        s.t. Σ wᵢ = 1
             w_min ≤ wᵢ ≤ w_max
             wᵢ ≤ risky_max  si actif risqué

    Avantage clé : ne nécessite aucune estimation des rendements futurs.
    Seule la matrice de covariance Σ est utilisée.

    Utilise scipy.optimize.minimize avec la méthode SLSQP
    (Sequential Least Squares Programming) — adaptée aux problèmes
    quadratiques avec contraintes linéaires.
    """
    n          = len(assets)
    profil     = user_profile.get("profil", "Équilibré")
    constraints_cfg = PROFILE_CONSTRAINTS.get(profil, PROFILE_CONSTRAINTS["Équilibré"])

    w_min     = constraints_cfg["w_min"]
    w_max     = constraints_cfg["w_max"]
    risky_max = constraints_cfg["risky_max"]
    btc_max   = constraints_cfg["btc_max"]

    # Fonction objectif : variance du portefeuille
    def portfolio_variance(w):
        return float(w @ cov_matrix @ w)

    # Gradient analytique (accélère la convergence)
    def gradient(w):
        return 2 * cov_matrix @ w

    # Contrainte : somme des poids = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bornes par actif
    bounds = []
    for asset in assets:
        if asset == "Bitcoin":
            bounds.append((0.0, btc_max))
        elif asset in RISKY_ASSETS:
            bounds.append((w_min, min(w_max, risky_max)))
        else:
            bounds.append((w_min, w_max))

    # Point de départ : equal weight
    w0 = np.ones(n) / n

    result = minimize(
        fun         = portfolio_variance,
        x0          = w0,
        jac         = gradient,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if result.success:
        w = np.maximum(result.x, 0)   # Éliminer les poids négatifs numériques
        w = w / w.sum()               # Renormaliser
        logger.info("[Agent 5] MVP convergé — variance=%.4f", portfolio_variance(w))
        return w
    else:
        logger.warning("[Agent 5] MVP n'a pas convergé : %s — fallback equal weight", result.message)
        return _equal_weight(assets)


# ─── 2C. Robust Optimization ─────────────────────────────────────────────────

def _robust_optimization(
    cov_matrix   : np.ndarray,
    assets       : list[str],
    user_profile : dict,
    mean_returns : np.ndarray,
    kappa        : float = 1.0,
) -> np.ndarray:
    """
    Robust Optimization — méthode avancée.

    Résout le problème de Markowitz robuste :
        max  w^T μ - κ √(w^T Σ w)
         w
        s.t. Σ wᵢ = 1, wᵢ ≥ 0

    Interprétation :
        κ = paramètre d'aversion à l'incertitude
        κ faible → proche Markowitz classique (maximise le rendement)
        κ élevé  → proche MVP (minimise le risque)

    κ est dérivé du profil utilisateur :
        Prudent    → κ = 2.0 (très conservateur)
        Équilibré  → κ = 1.0 (équilibré)
        Dynamique  → κ = 0.3 (orienté rendement)

    Implémentation via scipy (sans cvxpy pour simplicité et portabilité).
    Le problème est reformulé en minimisation : min -objectif.
    """
    n = len(assets)

    def neg_robust_objective(w):
        ret  = float(w @ mean_returns)
        risk = float(np.sqrt(max(w @ cov_matrix @ w, 1e-10)))
        return -(ret - kappa * risk)   # Minimiser le négatif = maximiser

    def gradient_neg(w):
        cov_w   = cov_matrix @ w
        risk    = float(np.sqrt(max(w @ cov_matrix @ w, 1e-10)))
        d_ret   = mean_returns
        d_risk  = cov_w / risk if risk > 1e-10 else np.zeros(n)
        return -(d_ret - kappa * d_risk)

    profil          = user_profile.get("profil", "Équilibré")
    constraints_cfg = PROFILE_CONSTRAINTS.get(profil, PROFILE_CONSTRAINTS["Équilibré"])
    w_min           = constraints_cfg["w_min"]
    w_max           = constraints_cfg["w_max"]
    risky_max       = constraints_cfg["risky_max"]
    btc_max         = constraints_cfg["btc_max"]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    bounds = []
    for asset in assets:
        if asset == "Bitcoin":
            bounds.append((0.0, btc_max))
        elif asset in RISKY_ASSETS:
            bounds.append((w_min, min(w_max, risky_max)))
        else:
            bounds.append((w_min, w_max))

    # Plusieurs points de départ pour éviter les minima locaux
    best_result = None
    best_val    = np.inf

    for w0 in [np.ones(n) / n, _minimum_variance(cov_matrix, assets, user_profile)]:
        result = minimize(
            fun         = neg_robust_objective,
            x0          = w0,
            jac         = gradient_neg,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        if result.success and result.fun < best_val:
            best_val    = result.fun
            best_result = result

    if best_result is not None and best_result.success:
        w = np.maximum(best_result.x, 0)
        w = w / w.sum()
        logger.info("[Agent 5] Robust Optim convergée — κ=%.2f", kappa)
        return w
    else:
        logger.warning("[Agent 5] Robust Optim échouée — fallback MVP")
        return _minimum_variance(cov_matrix, assets, user_profile)


# ─── 3. Ajustement tactique macro + sentiment ─────────────────────────────────

def _tactical_adjustment(
    weights      : np.ndarray,
    assets       : list[str],
    macro_scores : dict,
    sent_scores  : dict,
    horizon      : str,
) -> np.ndarray:
    """
    Ajuste tactiquement les poids optimisés selon les scores macro et sentiment.

    Logique :
        w_final_i = w_optim_i × (1 + tactical_weight × signal_i)

    où signal_i = score_macro_i × macro_weight + score_sentiment_i × sent_weight

    Plus l'horizon est long, plus l'ajustement tactique est réduit
    (le sentiment est non pertinent sur 10 ans).

    Les poids sont renormalisés après ajustement pour sommer à 1.
    """
    tac_weight = TACTICAL_WEIGHT.get(horizon, 0.15)

    if tac_weight == 0:
        return weights

    adjusted = weights.copy()
    for i, asset in enumerate(assets):
        macro_s = macro_scores.get(asset, 0.0)
        sent_s  = sent_scores.get(asset, 0.0)

        # Signal combiné : 60% macro + 40% sentiment
        signal = 0.60 * macro_s + 0.40 * sent_s

        # Ajustement multiplicatif borné pour éviter les déviations extrêmes
        adjustment = 1.0 + tac_weight * np.clip(signal, -1.0, 1.0)
        adjusted[i] = weights[i] * adjustment

    # Renormaliser
    total = adjusted.sum()
    if total > 1e-10:
        adjusted = adjusted / total

    logger.info("[Agent 5] Ajustement tactique appliqué (tac_weight=%.2f, horizon=%s)",
                tac_weight, horizon)
    return adjusted


# ─── 4. Application des contraintes finales ───────────────────────────────────

def _apply_final_constraints(
    weights      : np.ndarray,
    assets       : list[str],
    user_profile : dict,
) -> np.ndarray:
    """
    Applique les contraintes min/max finales après l'ajustement tactique.
    Utilise un algorithme de projection itératif simple.
    """
    profil          = user_profile.get("profil", "Équilibré")
    constraints_cfg = PROFILE_CONSTRAINTS.get(profil, PROFILE_CONSTRAINTS["Équilibré"])
    w_min           = constraints_cfg["w_min"]
    w_max           = constraints_cfg["w_max"]
    risky_max       = constraints_cfg["risky_max"]
    btc_max         = constraints_cfg["btc_max"]

    w = weights.copy()

    for iteration in range(50):
        # Clipper chaque poids selon ses bornes
        for i, asset in enumerate(assets):
            if asset == "Bitcoin":
                w[i] = np.clip(w[i], 0.0, btc_max)
            elif asset in RISKY_ASSETS:
                w[i] = np.clip(w[i], w_min, min(w_max, risky_max))
            else:
                w[i] = np.clip(w[i], w_min, w_max)

        # Renormaliser
        total = w.sum()
        if total > 1e-10:
            w = w / total

        # Vérifier convergence
        if abs(total - 1.0) < 1e-8:
            break

    return w


# ─── 5. Métriques du portefeuille final ───────────────────────────────────────

def _portfolio_metrics(
    weights      : np.ndarray,
    assets       : list[str],
    cov_matrix   : np.ndarray,
    mean_returns : np.ndarray,
    rf_annual    : float = 0.043,
) -> dict:
    """Calcule les métriques clés du portefeuille optimisé."""
    port_var     = float(weights @ cov_matrix @ weights)
    port_vol     = float(np.sqrt(port_var))
    port_ret     = float(weights @ mean_returns)
    sharpe       = (port_ret - rf_annual) / port_vol if port_vol > 0 else 0.0

    # Diversification Ratio
    asset_vols   = np.sqrt(np.diag(cov_matrix))
    weighted_vol = float(weights @ asset_vols)
    div_ratio    = weighted_vol / port_vol if port_vol > 0 else 1.0

    return {
        "volatility_annual"  : round(port_vol * 100, 2),
        "return_annual"      : round(port_ret * 100, 2),
        "sharpe_ratio"       : round(sharpe, 3),
        "diversification_ratio": round(div_ratio, 3),
        "max_weight"         : round(float(weights.max()) * 100, 1),
        "min_weight"         : round(float(weights[weights > 0.001].min()) * 100, 1),
    }


# ─── 6. Génération de la recommandation (Claude Sonnet) ──────────────────────

def _generate_recommendation(
    assets       : list[str],
    weights_dict : dict[str, float],
    metrics      : dict,
    user_profile : dict,
    macro_context: dict,
    sent_context : dict,
    method_used  : str,
    kappa        : float,
) -> str:
    """
    Génère la recommandation finale en langage naturel via Claude Sonnet.
    """
    if not ANTHROPIC_API_KEY:
        return "Clé Anthropic manquante — recommandation textuelle non disponible."

    profil   = user_profile.get("profil", "Équilibré")
    horizon  = user_profile.get("horizon", "medium")
    goal     = user_profile.get("goal", "Équilibrer rendement et risque")

    # Résumé des scores macro
    macro_scores = macro_context.get("scores", {})
    macro_top    = sorted(macro_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    macro_bot    = sorted(macro_scores.items(), key=lambda x: x[1])[:3]

    # Résumé des scores sentiment
    sent_scores  = sent_context.get("scores", {}) if isinstance(sent_context, dict) else {}

    allocation_text = "\n".join(
        f"  • {asset}: {pct:.1f}%"
        for asset, pct in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        if pct > 0.5
    )

    system_prompt = """Tu es un Portfolio Manager Senior dans une banque d'investissement.
Tu rédiges des recommandations claires, professionnelles et personnalisées pour tes clients.
Ton message doit être chaleureux, structuré, et justifier chaque choix d'allocation.
Réponds UNIQUEMENT en Markdown."""

    user_prompt = f"""
PROFIL DU CLIENT :
- Type : {profil}
- Horizon : {horizon}
- Objectif : {goal}
- Méthode d'optimisation utilisée : {method_used} (κ={kappa:.1f})

ALLOCATION FINALE ({len(assets)} actifs) :
{allocation_text}

MÉTRIQUES DU PORTEFEUILLE :
- Volatilité annualisée : {metrics['volatility_annual']}%
- Rendement attendu     : {metrics['return_annual']}%
- Ratio de Sharpe       : {metrics['sharpe_ratio']}
- Ratio de diversification : {metrics['diversification_ratio']}

CONTEXTE MACRO (top 3 positifs) :
{', '.join(f"{k} ({v:+.2f})" for k, v in macro_top)}

CONTEXTE MACRO (top 3 négatifs) :
{', '.join(f"{k} ({v:+.2f})" for k, v in macro_bot)}

SENTIMENT (résumé) :
Score moyen : {sent_context.get('average_score', 0) if isinstance(sent_context, dict) else 0:.2f}

RÉDIGE :
1. Une introduction personnalisée (profil + contexte de marché)
2. Le tableau d'allocation avec justification pour chaque position importante
3. Les risques principaux à surveiller
4. Une conclusion motivante adaptée au profil

Sois précis, professionnel et accessible. Évite le jargon inutile."""

    try:
        client   = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model       = "claude-sonnet-4-6",
            max_tokens  = 1500,
            temperature = 0.2,
            system      = system_prompt,
            messages    = [{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    except Exception as e:
        logger.error("[Agent 5] Claude Sonnet erreur : %s", e)
        return f"Erreur génération recommandation : {e}"


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_portfolio_optimization(state: dict) -> dict:
    """
    Pipeline complet d'optimisation de portefeuille.

    Étapes :
        1. Extraction des données depuis l'état LangGraph
        2. Filtrage des actifs éligibles (KYC)
        3. Optimisation MVP + Robust Optimization
        4. Ajustement tactique (macro + sentiment)
        5. Contraintes finales (profil)
        6. Recommandation Claude Sonnet
    """
    # ── Extraction des données ────────────────────────────────────────────────
    user_profile  = state.get("user_profile", {})
    macro_context = state.get("macro_context", {})
    sent_context  = state.get("sentiment_score", {})
    technical     = state.get("technical_risk", {})
    assets        = state.get("assets", [])

    profil  = user_profile.get("profil", "Équilibré")
    horizon = user_profile.get("horizon", "medium")

    # κ selon le profil (aversion à l'incertitude pour Robust Optim)
    kappa_map = {"Prudent": 2.0, "Équilibré": 1.0, "Dynamique": 0.3}
    kappa     = kappa_map.get(profil, 1.0)

    # ── Récupération de la matrice de covariance ──────────────────────────────
    cov_list = technical.get("covariance_matrix")
    tickers  = technical.get("tickers", assets)

    if cov_list is None or not tickers:
        logger.warning("[Agent 5] Matrice de covariance absente — fallback equal weight.")
        n        = len(assets)
        cov      = np.eye(n) * 0.04   # Volatilité 20% annuelle par défaut
        tickers  = assets
    else:
        cov = np.array(cov_list)

    # ── Rendements moyens historiques ─────────────────────────────────────────
    ann_returns = technical.get("annual_returns", {})
    if ann_returns and tickers:
        mean_ret = np.array([ann_returns.get(t, 0.05) for t in tickers])
    else:
        mean_ret = np.ones(len(tickers)) * 0.05   # 5% par défaut

    # ── Filtrage des actifs éligibles ─────────────────────────────────────────
    eligible_assets, sub_cov = _filter_eligible_assets(
        assets, cov, tickers, user_profile
    )
    n = len(eligible_assets)

    # Sous-vecteur des rendements pour les actifs éligibles
    eligible_indices = [tickers.index(a) for a in eligible_assets if a in tickers]
    sub_ret = mean_ret[eligible_indices] if eligible_indices else np.ones(n) * 0.05

    # ── Optimisation ──────────────────────────────────────────────────────────
    print(f"\n[Agent 5] Optimisation sur {n} actifs éligibles")
    print(f"[Agent 5] Profil : {profil} | κ = {kappa} | Horizon : {horizon}")

    # Baseline : Equal Weight
    w_ew  = _equal_weight(eligible_assets)
    var_ew = float(w_ew @ sub_cov @ w_ew)

    # Méthode principale : Minimum Variance
    w_mvp  = _minimum_variance(sub_cov, eligible_assets, user_profile)
    var_mvp = float(w_mvp @ sub_cov @ w_mvp)

    # Méthode avancée : Robust Optimization
    w_robust = _robust_optimization(sub_cov, eligible_assets, user_profile,
                                     sub_ret, kappa=kappa)
    var_robust = float(w_robust @ sub_cov @ w_robust)

    # Rapport de comparaison
    print(f"\n[Agent 5] {'Méthode':<22} {'Variance':>10} {'Vol ann.':>10}")
    print(f"[Agent 5] {'─'*44}")
    for name, w, var in [("equal_weight", w_ew, var_ew),
                          ("min_variance", w_mvp, var_mvp),
                          ("robust_optim", w_robust, var_robust)]:
        print(f"[Agent 5] {name:<22} {var:>10.4f} {np.sqrt(var)*100:>9.2f}%")

    # Sélection de la méthode : MVP si robust ≈ MVP, sinon Robust
    if abs(var_robust - var_mvp) / max(var_mvp, 1e-10) < 0.05:
        # Les deux convergent vers la même solution → MVP
        w_final   = w_mvp
        method_used = "Minimum Variance"
        print(f"\n[Agent 5] → Méthode retenue : Minimum Variance (Robust ≈ MVP)")
    elif var_robust < var_mvp:
        # Robust donne moins de variance → meilleure solution
        w_final   = w_robust
        method_used = "Robust Optimization"
        print(f"\n[Agent 5] → Méthode retenue : Robust Optimization")
    else:
        w_final   = w_mvp
        method_used = "Minimum Variance"
        print(f"\n[Agent 5] → Méthode retenue : Minimum Variance (plus stable)")

    # ── Ajustement tactique ───────────────────────────────────────────────────
    macro_scores = macro_context.get("scores", {}) if isinstance(macro_context, dict) else {}
    sent_scores  = sent_context.get("scores", {})  if isinstance(sent_context, dict)  else {}

    w_tactical = _tactical_adjustment(
        w_final, eligible_assets, macro_scores, sent_scores, horizon
    )

    # ── Contraintes finales ───────────────────────────────────────────────────
    w_constrained = _apply_final_constraints(w_tactical, eligible_assets, user_profile)

    # ── Métriques finales ─────────────────────────────────────────────────────
    metrics = _portfolio_metrics(w_constrained, eligible_assets,
                                  sub_cov, sub_ret)

    # ── Construction du dictionnaire d'allocation (en %) ─────────────────────
    allocation = {
        asset: round(float(w_constrained[i]) * 100, 2)
        for i, asset in enumerate(eligible_assets)
    }

    print(f"\n[Agent 5] Allocation finale :")
    for asset, pct in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(pct / 2)
        print(f"  {asset:<26} {pct:>6.2f}%  {bar}")
    print(f"\n[Agent 5] Vol={metrics['volatility_annual']}% | "
          f"Ret={metrics['return_annual']}% | "
          f"Sharpe={metrics['sharpe_ratio']} | "
          f"DR={metrics['diversification_ratio']}")

    # ── Recommandation Claude Sonnet ──────────────────────────────────────────
    recommendation = _generate_recommendation(
        assets        = eligible_assets,
        weights_dict  = allocation,
        metrics       = metrics,
        user_profile  = user_profile,
        macro_context = macro_context if isinstance(macro_context, dict) else {},
        sent_context  = sent_context  if isinstance(sent_context, dict)  else {},
        method_used   = method_used,
        kappa         = kappa,
    )

    return {
        "portfolio_allocation": allocation,
        "final_recommendation": recommendation,
        "optimization_details": {
            "method_used"         : method_used,
            "kappa"               : kappa,
            "eligible_assets"     : eligible_assets,
            "metrics"             : metrics,
            "weights_ew"          : dict(zip(eligible_assets, (w_ew * 100).round(2))),
            "weights_mvp"         : dict(zip(eligible_assets, (w_mvp * 100).round(2))),
            "weights_robust"      : dict(zip(eligible_assets, (w_robust * 100).round(2))),
            "variance_ew"         : round(var_ew, 6),
            "variance_mvp"        : round(var_mvp, 6),
            "variance_robust"     : round(var_robust, 6),
        },
    }


# ─── Nœud LangGraph ──────────────────────────────────────────────────────────

def portfolio_node(state: AgentState) -> dict:
    """Agent 5 — Nœud LangGraph."""
    print("─" * 60)
    print("  [Agent 5] Portfolio Manager — Optimisation")
    print("─" * 60)
    return run_portfolio_optimization(state)


# ─── Test local ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    np.random.seed(42)
    n = 10
    assets = [
        "Nasdaq-100", "MSCI EM IMI", "STOXX Eur600 Finance",
        "Global Healthcare", "US Treasury 10+Y", "Bund 1-3Y",
        "Clean Energy", "Global REITs", "Gold", "Bitcoin",
    ]

    # Matrice de covariance simulée réaliste
    A   = np.random.randn(n, n) * 0.1
    cov = A @ A.T + np.diag([0.21, 0.18, 0.25, 0.16, 0.14,
                               0.012, 0.29, 0.18, 0.16, 0.75]) ** 2

    test_state = {
        "assets"      : assets,
        "user_profile": {
            "profil"    : "Équilibré",
            "risk_score": 5,
            "horizon"   : "medium",
            "goal"      : "Équilibrer rendement et risque",
            "known_assets": assets,
        },
        "macro_context": {
            "scores": {a: np.random.uniform(-0.5, 0.5) for a in assets}
        },
        "sentiment_score": {
            "scores": {a: np.random.uniform(-0.3, 0.3) for a in assets},
            "average_score": 0.05,
        },
        "technical_risk": {
            "covariance_matrix": cov.tolist(),
            "tickers"          : assets,
            "annual_returns"   : {a: np.random.uniform(0.03, 0.20) for a in assets},
        },
    }

    result = run_portfolio_optimization(test_state)
    print("\n✅ Test terminé")
    print("Méthode :", result["optimization_details"]["method_used"])