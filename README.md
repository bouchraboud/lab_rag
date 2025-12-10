# RAG Ollama Lab - IPCC AR6

## Description
Application RAG (Retrieval-Augmented Generation) pour interroger les rapports IPCC AR6.

## Technologies
- Ollama (LLMs locaux)
- LangChain (pipeline RAG)
- ChromaDB (base vectorielle)
- FastAPI (API backend)
- Streamlit (interface utilisateur)

## Installation

1. Installer Ollama et télécharger les modèles:
```bash
ollama serve &
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

2. Créer l'environnement virtuel:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Télécharger les PDFs dans data/

## Utilisation

1. Ingestion des PDFs:
```bash
python ingest.py
```

2. Création des embeddings:
```bash
python embeddings.py
```

3. Lancer l'API:
```bash
uvicorn app:app --reload --port 8000
```

4. Lancer l'interface (nouveau terminal):
```bash
streamlit run ui_streamlit.py
```

## Choix de conception

**Chunk size:** 1000 caractères avec overlap de 200
- Permet de capturer le contexte nécessaire
- Balance entre précision et performance

**Modèle d'embeddings:** nomic-embed-text
- Optimisé pour la recherche sémantique
- Performant et rapide localement

**Retriever:** Similarité top-4
- Équilibre entre contexte et bruit
- Testé empiriquement

## Exemples de requêtes

1. "What are the main findings about global warming?"
2. "What is the projected sea level rise by 2100?"
3. "How does climate change affect biodiversity?"

