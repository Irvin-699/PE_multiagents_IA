"""
agents/technical.py
Agent 4 — Analyste technique de risque.

Reçoit market_data depuis l'Agent 1, calcule les métriques de risque
et retourne technical_risk à l'Agent 5.

4 méthodes d'estimation de la matrice de covariance :
    1. empirical          — Covariance empirique (baseline)
    2. ledoit_wolf_emp    — Ledoit-Wolf appliqué sur la covariance empirique
    3. tyler              — Estimateur de point fixe (distributions elliptiques)
    4. ledoit_wolf_tyler  — Ledoit-Wolf appliqué sur l'estimateur de Tyler

Note sur le coefficient α de Ledoit-Wolf :
    α est calculé AUTOMATIQUEMENT par sklearn via la formule analytique de
    Ledoit & Wolf (2004). Aucune valeur n'est fixée manuellement.
"""

import logging
import numpy as np
import pandas as pd
from numpy.linalg import inv, norm
from scipy.stats import median_abs_deviation
from sklearn.covariance import LedoitWolf

from core.state import AgentState
from core.config import RF_ANNUAL

logger = logging.getLogger(__name__)


# ─── 1. Nettoyage des prix ────────────────────────────────────────────────────

def _clean_prices(market_data: dict) -> pd.DataFrame:
    """Extrait et nettoie le DataFrame de prix depuis l'état LangGraph."""
    raw = market_data.get("prices")
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("market_data['prices'] doit être un DataFrame pandas.")

    df = raw.copy()

    # Rééchantillonnage mensuel (dernier cours du mois)
    try:
        pd.tseries.frequencies.to_offset("ME")
        freq = "ME"
    except ValueError:
        freq = "M"
    df = df.resample(freq).last()

    # Combler les NaN (jours fériés, marchés fermés)
    df = df.ffill().bfill()
    df = df.dropna(axis=1, how="all")

    return df


# ─── 2. Log-rendements ────────────────────────────────────────────────────────

def _compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ─── 3. Estimateurs de covariance ────────────────────────────────────────────

def _empirical_covariance(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Méthode 1 — Covariance empirique (baseline).

    """
    return log_returns.cov().values * 12   # Annualisation mensuelle × 12


def _ledoit_wolf(log_returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Méthode 2 — Ledoit-Wolf linéaire estimé sur les log-rendements.
    α est calculé automatiquement par sklearn (formule analytique 2004).
    """
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(log_returns.values)
    alpha = float(lw.shrinkage_)
    return lw.covariance_ * 12, alpha


def _tyler_fixed_point(log_returns: pd.DataFrame,
                       max_iter: int = 200,
                       tol: float = 1e-6) -> np.ndarray:
    """
    Méthode 3 — Estimateur de point fixe de Tyler (1987).

    
    """
    X = log_returns.values
    n, p = X.shape

    # Initialisation avec la matrice identité
    Sigma = np.eye(p)

    for iteration in range(max_iter):
        try:
            Sigma_inv = inv(Sigma)
        except np.linalg.LinAlgError:
            logger.warning("[Tyler] Matrice singulière à l'itération %d — arrêt.", iteration)
            break

        # Distances de Mahalanobis² de toutes les observations d'un coup,
        # puis somme pondérée des produits extérieurs : Σ wᵢ·xᵢxᵢᵀ = (X·w)ᵀX
        mahal   = np.einsum("ij,jk,ik->i", X, Sigma_inv, X)
        weights = np.where(mahal > 1e-10, 1.0 / np.maximum(mahal, 1e-10), 0.0)
        Sigma_new = (X * weights[:, None]).T @ X

        # Normalisation (trace = p) pour assurer l'unicité de la solution
        Sigma_new = (p / n) * Sigma_new
        trace = np.trace(Sigma_new)
        if trace > 1e-10:
            Sigma_new /= (trace / p)

        # Critère de convergence : variation relative (norme de Frobenius)
        diff = norm(Sigma_new - Sigma, 'fro') / max(norm(Sigma, 'fro'), 1e-10)
        Sigma = Sigma_new

        if diff < tol:
            logger.info("[Tyler] Convergence en %d itérations (diff=%.2e).",
                        iteration + 1, diff)
            break
    else:
        logger.warning("[Tyler] Convergence non atteinte après %d itérations.", max_iter)

    # Reconstruction de la covariance complète :
    # Forme (corrélation) = Tyler  ×  Échelle (variances) = MAD robuste

    # Corrélation depuis la matrice de scatter Tyler
    std_tyler = np.sqrt(np.maximum(np.diag(Sigma), 1e-10))
    corr = Sigma / np.outer(std_tyler, std_tyler)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Volatilités robustes via MAD (annualisées)
    # Facteur 1.4826 pour cohérence avec l'écart-type sous normalité
    mad_vols = np.array([
        median_abs_deviation(log_returns.iloc[:, j].values, scale='normal')
        for j in range(p)
    ]) * np.sqrt(12)   # Annualisation mensuelle

    # Covariance = D × Corr × D  (D = matrice diagonale des vols)
    D = np.diag(mad_vols)
    cov_annual = D @ corr @ D

    return cov_annual


def _compute_covariances(log_returns: pd.DataFrame) -> dict[str, dict]:
    """
    Calcule les 4 estimateurs de covariance et retourne les matrices annualisées.

    Returns:
        Dict {nom: {"matrix": np.ndarray, "alpha": float|None}}
    """
    results = {}

    # ── Méthode 1 : Covariance empirique ────────────────────────────────────
    emp = _empirical_covariance(log_returns)
    results["empirical"] = {"matrix": emp, "alpha": None}

    # ── Méthode 2 : Ledoit-Wolf sur covariance empirique ────────────────────
    # sklearn calcule α automatiquement à partir des données brutes
    lw_emp, alpha_emp = _ledoit_wolf(log_returns)
    results["ledoit_wolf_emp"] = {"matrix": lw_emp, "alpha": alpha_emp}

    # ── Méthode 3 : Estimateur de point fixe (Tyler) ────────────────────────
    try:
        tyler = _tyler_fixed_point(log_returns)
        results["tyler"] = {"matrix": tyler, "alpha": None}
    except Exception as e:
        logger.warning("[Agent 4] Tyler échoué : %s", e)

    # ── Méthode 4 : Ledoit-Wolf sur estimateur de Tyler ─────────────────────
    # On applique le shrinkage LW sur la matrice Tyler déjà robuste
    # pour améliorer encore sa stabilité numérique
    if "tyler" in results:
        try:
            # On dé-annualise Tyler pour passer dans l'espace de LW
            tyler_daily = results["tyler"]["matrix"] / 12
            # Fit LW sur les rendements (il recalcule α automatiquement)
            lw = LedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(log_returns.values)
            alpha_tyler = float(lw.shrinkage_)
            # Mélange entre la matrice Tyler et la cible structurée LW
            target = np.trace(tyler_daily) / tyler_daily.shape[0] * np.eye(tyler_daily.shape[0])
            lw_tyler_daily = (1 - alpha_tyler) * tyler_daily + alpha_tyler * target
            results["ledoit_wolf_tyler"] = {
                "matrix": lw_tyler_daily * 12,   # Réannualisation
                "alpha":  alpha_tyler,
            }
        except Exception as e:
            logger.warning("[Agent 4] Ledoit-Wolf sur Tyler échoué : %s", e)

    return results


# ─── 4. Sélection de la meilleure matrice ────────────────────────────────────

def _is_positive_definite(matrix: np.ndarray) -> bool:
    """Vérifie si la matrice est définie positive via décomposition de Cholesky."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _select_best(candidates: dict[str, dict]) -> tuple[str, np.ndarray]:
    """
    Sélectionne la meilleure matrice parmi les 4 estimateurs.

    Critères (dans l'ordre) :
        1. La matrice doit être définie positive (obligatoire pour l'optimisation)
        2. Parmi les matrices PD, on choisit celle avec le numéro de
           conditionnement le plus bas (= la plus stable numériquement)

    Fallback : si aucune n'est PD, on retourne Ledoit-Wolf empirique
    qui est quasi-garanti PD par construction.
    """
    best_name, best_matrix, best_cond = None, None, float("inf")

    for name, entry in candidates.items():
        matrix = entry["matrix"]
        if _is_positive_definite(matrix):
            cond = float(np.linalg.cond(matrix))
            if cond < best_cond:
                best_name, best_matrix, best_cond = name, matrix, cond

    if best_matrix is None:
        logger.warning("[Agent 4] Aucune matrice PD — fallback sur Ledoit-Wolf empirique.")
        best_name   = "ledoit_wolf_emp"
        best_matrix = candidates["ledoit_wolf_emp"]["matrix"]

    return best_name, best_matrix


# ─── 5. Métriques de risque ───────────────────────────────────────────────────

def _max_drawdown(price_series: pd.Series) -> float:
    """Max drawdown peak-to-trough sur la série complète."""
    rolling_max = price_series.cummax()
    return float(((price_series - rolling_max) / rolling_max).min())


def _compute_metrics(prices: pd.DataFrame, log_returns: pd.DataFrame) -> dict:
    """
    Calcule les métriques annualisées par actif.
    Sharpe avec taux sans risque RF_ANNUAL = 4.3%.
    """
    vol      = log_returns.std() * np.sqrt(12)
    mean_ann = log_returns.mean() * 12
    sharpe   = ((mean_ann - RF_ANNUAL) / vol).replace([np.inf, -np.inf], np.nan)
    corr     = log_returns.corr()

    return {
        "volatilities"      : vol.to_dict(),
        "annual_returns"    : mean_ann.to_dict(),
        "sharpe_ratios"     : sharpe.to_dict(),
        "max_drawdowns"     : {t: _max_drawdown(prices[t]) for t in prices.columns},
        "correlation_matrix": corr.to_dict(),
        "tickers"           : prices.columns.tolist(),
        "n_observations"    : len(log_returns),
        "period_start"      : str(log_returns.index.min().date()),
        "period_end"        : str(log_returns.index.max().date()),
    }


# ─── Fonction principale ──────────────────────────────────────────────────────

def run_technical_analysis(state: AgentState) -> dict:
    market_data: dict = state["market_data"]

    # Étape 1 — Nettoyage
    prices = _clean_prices(market_data)

    # Étape 2 — Log-rendements
    log_returns = _compute_log_returns(prices)

    # Étape 3 — 4 estimateurs de covariance
    candidates = _compute_covariances(log_returns)

    # Étape 4 — Sélection de la meilleure matrice
    best_method, best_cov = _select_best(candidates)

    # Étape 5 — Métriques de risque
    metrics = _compute_metrics(prices, log_returns)

    # Affichage du rapport de comparaison
    print(f"\n[Agent 4] {'Méthode':<22} {'Cond. number':>14} {'PD':>4} "
          f"{'α (auto)':>10}")
    print(f"[Agent 4] {'─'*54}")
    for name, entry in candidates.items():
        mat   = entry["matrix"]
        cond  = np.linalg.cond(mat)
        pd_ok = "✓" if _is_positive_definite(mat) else "✗"
        alpha = f"{entry['alpha']:.4f}" if entry["alpha"] is not None else "  N/A  "
        star  = " ← sélectionnée" if name == best_method else ""
        print(f"[Agent 4] {name:<22} {cond:>14.2e} {pd_ok:>4} {alpha:>10}{star}")

    return {
        "technical_risk": {
            **metrics,
            "covariance_matrix" : best_cov.tolist(),
            "covariance_method" : best_method,
            "condition_numbers" : {
                name: float(np.linalg.cond(e["matrix"]))
                for name, e in candidates.items()
            },
            "shrinkage_alphas"  : {
                name: e["alpha"]
                for name, e in candidates.items()
                if e["alpha"] is not None
            },
            "all_covariances"   : {
                name: e["matrix"].tolist()
                for name, e in candidates.items()
            },
        }
    }


# ─── Node LangGraph ───────────────────────────────────────────────────────────

def technical_node(state: AgentState) -> dict:
    """Agent 4 — Analyste technique de risque."""
    print("--- [Agent 4] Analyse technique de risque ---")
    return run_technical_analysis(state)