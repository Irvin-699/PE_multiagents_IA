from typing import TypedDict, Annotated, List, Dict, Any

class AgentState(TypedDict):
    """
    État global partagé entre les agents du système LangGraph.
    """
    # Données brutes et métadonnées
    assets: List[str]  # Liste des 10 actifs sélectionnés
    market_data: Dict[str, Any]  # Données brutes de prix
    
    # Analyses des agents
    macro_context: str  # Analyse fondamentale / Macro
    sentiment_score: Dict[str, float]  # Analyse sentimentale
    technical_risk: Dict[str, Any]  # Analyse technique de risque (volatilité, corrélations)
    
    # Décision du portfolio manager
    portfolio_allocation: Dict[str, float]  # Pourcentage d'allocation finale
    final_recommendation: str  # Recommandation pour l'interface
