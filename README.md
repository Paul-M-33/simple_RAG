# RAG — Question/Réponse sur documents PDF

Agent RAG (Retrieval Augmented Generation) permettant d'interroger des documents PDF en langage naturel, propulsé par Gemini et LangChain.

## Prérequis

- Python 3.10+
- Une clé API Gemini — obtenir sur [aistudio.google.com](https://aistudio.google.com)

## Installation

```bash
pip install langchain langchain-core langchain-community langchain-google-genai langchain-chroma langchain-text-splitters chromadb pypdf python-dotenv
```

## Configuration

Crée un fichier `.env` à la racine du projet :

```
GOOGLE_API_KEY=ta_clé_gemini
```

## Structure du projet

```
rag_project/
├── .env              # clé API (ne pas commiter !)
├── ingest.py         # indexation des documents
├── query.py          # interface de questions
└── documents/        # place tes PDFs ici
```

## Utilisation

**1. Ajoute tes PDFs dans le dossier `documents/`**

**2. Lance l'indexation** (une seule fois, ou à chaque ajout de document)

```bash
python ingest.py
```

Cela crée un dossier `chroma_db/` contenant la base vectorielle.

**3. Lance l'interface de questions**

```bash
python query.py
```

Tu peux ensuite poser des questions en langage naturel sur tes documents. Tape `exit` pour quitter.

## Exemple

```
RAG prêt. Tape 'exit' pour quitter.

Question : Quelles sont les formations du candidat ?
Réponse : Le candidat a obtenu un Master 2 en Cryptologie et Sécurité Informatique
à l'Université de Bordeaux en 2017...

Question : exit
```

## Notes

- Le dossier `chroma_db/` n'a pas besoin d'être recréé à chaque lancement de `query.py`, uniquement quand tu modifies les documents.
- Ne commite jamais ton fichier `.env` — ajoute-le à ton `.gitignore`.