from langchain.embeddings.base import Embeddings
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv(dotenv_path="C:/Users/rk39331/source/repos/ai-core-projekt/aiprojekt/cred.env")

# Dokumente laden
loader = DirectoryLoader("C:/Users/rk39331/source/repos/ai-core-projekt/aiprojekt/data/information/", glob="*.md")
documents = loader.load()

# Dokumente in Chunks aufteilen
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=400, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# Embedding-Modell definieren und Datenbank erstellen
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-3-small')
db = Chroma.from_documents(texts, embedding_model)

# Abfrage-Retriever erstellen
retriever = db.as_retriever()

# ChatLLM erstellen
chat_llm = ChatOpenAI(proxy_model_name='gpt-4o-mini')

# QA-Instanz erstellen
qa = RetrievalQA.from_llm(llm=chat_llm, retriever=retriever)

# Beispielabfrage
query = input("prompt:")
print('\nRESPONSE:')
print(qa.invoke(query)['result'])


# Alle gespeicherten Dokumente ausgeben
print("\nAlle gespeicherten Dokumente in der Datenbank:")
for i, text in enumerate(texts):
    print(f"Dokument {i + 1}: {text.page_content}")
