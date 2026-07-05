# FinAgent P26 — Système Multi-Agents de Conseil en Investissement

Projet académique P26 : pipeline d'analyse financière piloté par des agents IA spécialisés, avec interface KYC Streamlit.

## Architecture

```
main.py          — Orchestrateur async (LangGraph)
app.py           — Interface Streamlit (questionnaire KYC 10 questions)
agents/
  fundamental.py — Agent 1 : analyse fondamentale des ETF
  market_data.py — Agent 2 : données de marché (Yahoo Finance)
  sentiment.py   — Agent 3 : analyse de sentiment (FinBERT)
  technical.py   — Agent 4 : indicateurs techniques
  portfolio.py   — Agent 5 : optimisation de portefeuille (PyPortfolioOpt)
core/
  state.py       — État partagé entre agents (AgentState)
  config.py      — Tickers & paramètres globaux
data/            — Base vectorielle, modèles, données historiques
```

## Pipeline

1. **Agent 1 (Fondamental)** — analyse qualitative des actifs éligibles
2. **Agents 2, 3, 4 (Parallèles)** — données marché + sentiment + technique via `asyncio.gather`
3. **Agent 5 (Portfolio)** — optimisation Markowitz sur l'univers filtré
4. **Interface KYC** — profil investisseur (horizon, risque, objectifs, ESG…) → recommandation personnalisée

Temps d'exécution estimé : **80–120 s** (vs ~900 s séquentiel).

## Univers d'investissement

| Nom | Ticker | Catégorie |
|-----|--------|-----------|
| Nasdaq-100 | ^NDX | Actions US tech |
| MSCI EM IMI | EIMI.L | Actions émergentes |
| STOXX Eur600 Finance | EXH2.DE | Actions Europe finance |
| Global Healthcare | IXJ | Actions santé |
| US Treasury 10+Y | LUTR.L | Obligations LT |
| Bund 1-3Y | DBXP.DE | Obligations CT |
| Clean Energy | ICLN | ESG énergie |
| Global REITs | EPRA.PA | Immobilier |
| Gold | GLD | Or |
| Bitcoin | BTC-USD | Crypto |

Période d'analyse : 2016-01-01 → 2025-12-31

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements_PETEM.txt
```

Créer un fichier `.env` à la racine :

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Lancement

```bash
# Interface Streamlit
python -m streamlit run app.py

# Pipeline en ligne de commande
python main.py
```

## Stack technique

- **Orchestration** : LangGraph, asyncio
- **LLM** : Claude (Anthropic), GPT (OpenAI)
- **Finance** : yfinance, PyPortfolioOpt, pandas, numpy
- **NLP** : FinBERT (sentiment financier)
- **Interface** : Streamlit
- **Vectoriel** : base vectorielle locale (`data/vector_db.py`)
