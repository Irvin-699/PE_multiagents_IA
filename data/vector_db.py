import os
import chromadb
from chromadb.config import Settings

def get_chroma_client():
    """
    Initialise et retourne le client local ChromaDB.
    Les données vectorielles seront stockées dans le dossier local 'data/chroma_db'.
    """
    # Définition du chemin de stockage local (relatif à ce script ou absolu)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "chroma_db")
    
    # Création du client persistant
    # Chroma 0.4.x utilise PersistentClient
    client = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB initialisé. Emplacement: {db_path}")
    
    return client

def init_collections():
    """
    Crée ou récupère les collections nécessaires pour le projet.
    """
    client = get_chroma_client()
    
    # Collection pour les actualités
    news_collection = client.get_or_create_collection(name="financial_news")
    print(f"Collection 'financial_news' vérifiée/créée. Total d'éléments : {news_collection.count()}")
    
    return news_collection

if __name__ == "__main__":
    # Test d'initialisation
    init_collections()
