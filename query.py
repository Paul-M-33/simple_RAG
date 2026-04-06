from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Recharger la base vectorielle
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Prompt

prompt = PromptTemplate.from_template("""
Tu es un assistant qui répond uniquement à partir du contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}
Réponse :
""")

# 4. LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5. Chaîne LCEL (nouvelle syntaxe)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Boucle de questions
print("RAG prêt. Tape 'exit' pour quitter.\n")
while True:
    question = input("Question : ")
    if question.lower() == "exit":
        break

    response = chain.invoke(question)
    print(f"\nRéponse : {response}\n")