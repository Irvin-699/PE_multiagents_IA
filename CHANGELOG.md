# Changelog

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format s'appuie sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/)
et le projet suit un versionnage [SemVer](https://semver.org/lang/fr/).

## [1.0.0] — 2026-07-07

Refonte majeure : centralisation de la configuration, parallélisation des
agents 2 et 3, singleton FinBERT et nouvelle interface Streamlit (Wisevest).

### Ajouté

- `core/models.py` — singleton FinBERT thread-safe, chargé paresseusement et
  partagé entre les Agents 2 et 3 (auparavant chargé deux fois : ~2 × 440 Mo
  de RAM et ~2 × 10 s de chargement).
- `.env.example` — modèle documenté des clés API requises (Anthropic,
  Finnhub, NewsAPI).
- `.streamlit/config.toml` — configuration du thème de l'application.
- `CHANGELOG.md` — ce fichier.

### Modifié

- `core/config.py` — devient la source unique de vérité : chemins absolus
  (`PROJECT_ROOT`, Parquet/CSV, `CHROMA_PATH`), univers `TICKERS`,
  `START_DATE`, date de fin dynamique `end_date()`, taux sans risque
  `RF_ANNUAL`, timeouts réseau (`HTTP_TIMEOUT`) et LLM (`LLM_TIMEOUT`).
- `main.py` — nouvel ordonnancement respectant les dépendances :
  Agent 1 → Agent 2 → (Agent 3 ∥ Agent 4) → Agent 5 (l'Agent 3 lit la base
  ChromaDB écrite par l'Agent 2). Préchauffage de FinBERT en tâche de fond,
  timeout par agent porté à 180 s, reconfiguration de la console Windows en
  UTF-8. Objectif : pipeline complet < 60 s.
- `agents/fundamental.py` (Agent 2) — réécriture asynchrone :
  `httpx.AsyncClient` + `AsyncAnthropic`, traitement concurrent des 10 ETF.
  Finnhub réduit à 1 appel global (au lieu de 10 appels espacés de 0,5 s),
  scoring FinBERT en un seul batch via le modèle partagé.
- `agents/sentiment.py` (Agent 3) — réécriture asynchrone ; réutilise les
  scores FinBERT déjà stockés dans ChromaDB (plus de re-scoring ni de second
  chargement du modèle), synthèses Claude Haiku en parallèle.
- `agents/technical.py` (Agent 4) — estimateur de Tyler vectorisé
  (`np.einsum` au lieu d'une boucle sur les observations), signature de
  Ledoit-Wolf simplifiée, `RF_ANNUAL` importé de la config.
- `agents/market_data.py` (Agent 1) — tickers, période et chemins importés
  de `core/config.py`, timeout ajouté aux téléchargements yfinance.
- `agents/portfolio.py` (Agent 5) — sélection de la méthode d'optimisation
  pilotée par le profil KYC : Prudent → Minimum Variance, Dynamique →
  Robust Optimization, Équilibré → comparaison des variances.
- `app.py` — refonte complète de l'interface Streamlit : assistant KYC en
  5 étapes, suivi visuel de l'exécution du pipeline, graphiques Plotly
  (donut d'allocation, barres divergentes de sentiment), cartes de métriques.
- `test_pipeline.py` — correction de l'encodage console Windows (UTF-8) qui
  faisait échouer le test sur cp1252.
- `requirements_PETEM.txt` — réencodé en UTF-8 (était en UTF-16), dépendances
  épinglées et alignées sur la refonte (httpx, anthropic, torch CPU…).
- `.gitignore` — ignore `data/models/` (cache FinBERT/ChromaDB) et
  ré-autorise `.env.example`.
- `README.md` — le projet devient **Wisevest** (ex-FinAgent P26).

### Supprimé

- `agents/app.py` — ancienne application Streamlit dupliquée (remplacée par
  `app.py` à la racine).
- `data/vector_db.py` — module ChromaDB inutilisé (le chemin est désormais
  centralisé dans `core/config.py`).
- `test_clé.py`, `test_google.py` — scripts de test ad hoc obsolètes.

## [0.1.0] — 2026-06-02

- Version initiale : pipeline multi-agents P26 complet (5 agents LangGraph),
  première application Streamlit de profil investisseur, README.
