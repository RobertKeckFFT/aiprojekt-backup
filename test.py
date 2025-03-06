from flask import Flask, request, jsonify
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

# Flask App initialisieren
app = Flask(__name__)

# Dokumente laden
loader = DirectoryLoader("C:/Users/rk39331/source/repos/ai-core-projekt/aiprojekt/data/information/", glob="*.md")
documents = loader.load()

# Dokumente in Chunks aufteilen
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=10)
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

@app.route("/query", methods=["POST"])
def query_llm():
    """Verarbeitet Anfragen von SAP Build Apps"""
    data = request.json
    user_input = data.get("input")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    response = qa.invoke(user_input)

    return jsonify({"response": response['result']})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
