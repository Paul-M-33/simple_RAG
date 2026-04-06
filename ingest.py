from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Charger tous les PDFs du dossier documents/
loader = DirectoryLoader("documents/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
print(f"{len(docs)} pages chargées")

# 2. Découper en chunks
# chunk_size = taille max d'un morceau (en caractères)
# chunk_overlap = chevauchement entre chunks pour ne pas perdre le contexte aux jointures
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks créés")

# 3. Créer les embeddings et stocker dans Chroma (persisté sur disque dans ./chroma_db)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
print("Indexation terminée, base vectorielle sauvegardée dans ./chroma_db")