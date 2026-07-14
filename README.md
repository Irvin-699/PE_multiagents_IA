# FinAgent P26 — Système Multi-Agents de Conseil en Investissement

Système d'intelligence artificielle qui génère une recommandation de portefeuille personnalisée à partir du profil investisseur d'un utilisateur. Cinq agents spécialisés collaborent en pipeline pour analyser les marchés, l'actualité financière et le sentiment des investisseurs, puis optimisent mathématiquement l'allocation.

---

## Vue d'ensemble du système

```
Utilisateur (KYC Streamlit)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE MULTI-AGENTS                    │
│                                                             │
│  Agent 1            Agent 2            Agent 3 ──┐          │
│  Collecte           Macro +            Sentiment  │ parallèle│
│  des prix           ChromaDB           des invest.│          │
│  (Yahoo Finance)    (Finnhub/NewsAPI   (ChromaDB  │          │
│                      + FinBERT         + FinBERT  │          │
│                      + Claude Haiku)   + Cl. Haiku│          │
│       │                  │                  ─────┘          │
│       │             Agent 4 ──────────────────              │
│       │             Risque technique        │ parallèle      │
│       │             (covariance, Sharpe,    │                │
│       │              drawdown, Tyler)       │                │
│       └──────────────────┴─────────────────┘                │
│                          │                                  │
│                     Agent 5                                  │
│                     Portfolio Manager                        │
│                     (MVP + Robust Optim + Claude Sonnet)     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Recommandation personnalisée (Markdown)
```

---

## Fonctionnement étape par étape

### 1. Questionnaire KYC (interface Streamlit)

L'utilisateur répond à 10 questions via `app.py` :

- Horizon d'investissement (court / moyen / long / très long)
- Tolérance au risque (score 1–10)
- Objectif financier
- Connaissance des marchés et des classes d'actifs
- Préférences ESG

Ces réponses construisent un `user_profile` qui pilote toutes les décisions d'allocation (contraintes de poids, pénalisation des actifs risqués, ajustement tactique).

---

### Agent 1 — Collecte des données de marché

**Fichier :** `agents/market_data.py`

Charge l'historique de prix (2016 → aujourd'hui) depuis un fichier Parquet local, puis télécharge uniquement les jours ouvrés manquants via **Yahoo Finance** (`yfinance`). Le résultat est un DataFrame de prix journaliers pour les 10 ETF de l'univers d'investissement.

Les données sont transmises à tous les agents suivants via l'état partagé `AgentState`.

---

### Agent 2 — Analyste macro-économique

**Fichier :** `agents/fundamental.py`  
**LLM :** Claude Haiku

Pour chaque ETF, l'agent exécute un pipeline en 4 étapes :

1. **Collecte** — Articles de presse récents (7 derniers jours) depuis **Finnhub** et **NewsAPI**, filtrés par mots-clés spécifiques à chaque ETF (ex. `"Fed rate tech nasdaq"` pour le Nasdaq-100).

2. **Filtrage FinBERT** — Le modèle NLP financier **ProsusAI/FinBERT** score chaque titre : `positive → +score`, `negative → −score`, `neutral → score × 0.1`. Les articles sont triés par signal décroissant.

3. **Stockage ChromaDB** — Les 5 meilleurs articles sont vectorisés et persistés dans une base **ChromaDB** locale (`data/models/chroma`). L'Agent 3 relira cette base — c'est pourquoi l'Agent 2 doit impérativement terminer avant l'Agent 3.

4. **Analyse Claude Haiku** — Le LLM reçoit les 5 articles les plus significatifs et produit une analyse structurée avec un **score macro** entre −1.0 (très défavorable) et +1.0 (très favorable) pour chaque ETF.

Le score est pondéré par l'horizon d'investissement : un horizon court donne plus de poids à la macro (40%) qu'un horizon très long (5%).

---

### Agent 3 — Analyste sentimental

**Fichier :** `agents/sentiment.py`  
**LLM :** Claude Haiku

Mesure non pas les faits économiques (c'est l'Agent 2), mais **ce que les investisseurs pensent et ressentent**. S'exécute en parallèle de l'Agent 4 après la fin de l'Agent 2.

1. **Requête ChromaDB** — Récupère les articles de l'ETF déjà stockés par l'Agent 2, via une requête sémantique orientée sentiment.

2. **Filtrage par marqueurs linguistiques** — Conserve uniquement les articles contenant des marqueurs de peur, d'optimisme, d'anticipation ou d'incertitude (ex. `"investors fear"`, `"expect that"`, `"panic"`, `"bullish on"`). Si trop peu d'articles matchent, fallback sur les articles au score FinBERT absolu le plus élevé.

3. **Re-scoring FinBERT** — Nouveau passage FinBERT uniquement sur les articles sentimentaux filtrés, pour une mesure plus représentative de l'humeur de marché.

4. **Synthèse Claude Haiku** — Prompt orienté psychologie des marchés : humeur dominante, anticipations collectives, **score sentiment** entre −1.0 (panique) et +1.0 (euphorie).

Pondération par horizon : le sentiment est très influent à court terme (35%) et quasi ignoré au-delà de 10 ans (5%).

---

### Agent 4 — Analyste technique de risque

**Fichier :** `agents/technical.py`  
**Bibliothèques :** NumPy, SciPy, scikit-learn

S'exécute en parallèle de l'Agent 3. À partir des prix journaliers de l'Agent 1, calcule :

- **Log-rendements** mensuels sur l'ensemble de la période
- **4 estimateurs de matrice de covariance** comparés simultanément :

  | Méthode | Description |
  |---------|-------------|
  | Empirique | Baseline classique |
  | Ledoit-Wolf | Shrinkage linéaire automatique (α calculé analytiquement) |
  | Tyler | Estimateur robuste aux valeurs extrêmes (point fixe, pondération Mahalanobis) |
  | Ledoit-Wolf + Tyler | Shrinkage appliqué à la matrice Tyler |

  La matrice retenue est la **plus stable numériquement** (numéro de conditionnement minimal) parmi celles qui sont définies positives.

- **Métriques par actif** : volatilité annualisée, rendement moyen, ratio de Sharpe (taux sans risque = 4,3%), max drawdown, matrice de corrélation.

---

### Agent 5 — Portfolio Manager

**Fichier :** `agents/portfolio.py`  
**LLM :** Claude Sonnet

Agrège tous les signaux précédents et produit l'allocation finale en 5 étapes :

1. **Filtrage des actifs éligibles** selon le profil KYC (actifs connus de l'utilisateur, exclusion du Bitcoin pour profil Prudent, minimum 3 actifs).

2. **Optimisation mathématique** — Trois méthodes calculées et comparées :
   - *Equal Weight* (1/n) — baseline de référence
   - *Minimum Variance Portfolio* (MVP) — minimise `w^T Σ w` via SLSQP ; n'utilise pas les rendements futurs estimés, ce qui le rend plus robuste
   - *Robust Optimization* — maximise `w^T μ − κ√(w^T Σ w)` avec κ dérivé du profil (Prudent→2.0, Équilibré→1.0, Dynamique→0.3)

   Si les deux méthodes convergent vers une solution similaire (écart < 5%), MVP est retenu.

3. **Ajustement tactique** — Les poids optimisés sont modulés par les scores macro et sentiment :
   `w_final_i = w_optim_i × (1 + tac_weight × (0.6 × macro_i + 0.4 × sent_i))`
   L'ajustement est plus fort à court terme (30%) et réduit à long terme (5%).

4. **Application des contraintes finales** — Bornes min/max par actif selon le profil, renormalisées par projection itérative.

5. **Recommandation Claude Sonnet** — Le LLM reçoit l'allocation finale, les métriques du portefeuille, le résumé macro et sentiment, et rédige une recommandation professionnelle en Markdown, personnalisée au profil de l'utilisateur.

---

## Ordre d'exécution du pipeline

```
Étape 1 : Agent 1  (collecte des prix — bloquant, les autres en dépendent)
Étape 2 : Agent 2  (macro + ChromaDB — doit finir avant Agent 3)
Étape 3 : Agent 3 + Agent 4  (parallèle via asyncio.gather)
Étape 4 : Agent 5  (optimisation et recommandation finale)
```

Temps d'exécution estimé : **80–120 secondes** (vs ~900 s séquentiel).

---

## Univers d'investissement (10 ETF)

| ETF | Ticker | Catégorie | Disponible depuis |
|-----|--------|-----------|-------------------|
| Nasdaq-100 | ^NDX | Actions US tech | 1985 |
| MSCI EM IMI | EIMI.L | Actions émergentes | 2014 |
| STOXX Eur600 Finance | EXH2.DE | Actions Europe finance | 2001 |
| Global Healthcare | IXJ | Actions santé mondiale | 2001 |
| US Treasury 10+Y | LUTR.L | Obligations LT US | 2016 |
| Bund 1-3Y | DBXP.DE | Obligations CT Europe | 2007 |
| Clean Energy | ICLN | ESG / Énergie propre | 2008 |
| Global REITs | EPRA.PA | Immobilier mondial | 2016 |
| Gold | GLD | Or physique | 2004 |
| Bitcoin | BTC-USD | Crypto (satellite) | 2014 |

Période d'analyse commune : **2016-01-01 → 2025-12-31** (contrainte par les ETF les plus récents).

---

## Profils investisseur et contraintes

| Profil | Poids max/actif | Poids max actif risqué | Bitcoin | Sharpe prioritaire |
|--------|----------------|----------------------|---------|-------------------|
| Prudent | 20% | 10% | Non | Risque minimal |
| Équilibré | 25% | 20% | Max 5% | Équilibre rendement/risque |
| Dynamique | 35% | 35% | Max 15% | Rendement maximal |

---

## Stack technique

| Domaine | Technologies |
|---------|-------------|
| Orchestration | LangGraph, asyncio (pipeline parallèle) |
| LLM | Claude Haiku 4.5 (agents 2 & 3), Claude Sonnet 4.6 (agent 5) |
| Données marché | yfinance (Yahoo Finance), Finnhub API, NewsAPI |
| NLP financier | ProsusAI/FinBERT (HuggingFace Transformers) |
| Base vectorielle | ChromaDB (persistance locale) |
| Optimisation | SciPy SLSQP, NumPy, scikit-learn (Ledoit-Wolf) |
| Interface | Streamlit |
| Stockage | Parquet (pyarrow + snappy), CSV |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / Mac
pip install -r requirements_PETEM.txt
```

Créer un fichier `.env` à la racine :

```
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=...
NEWSAPI_KEY=...
```

---

## Lancement

```bash
# Interface Streamlit (recommandé)
python -m streamlit run app.py

# Pipeline en ligne de commande (profil par défaut : Équilibré)
python main.py
```
