from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi
import os
import configparser

config = configparser.ConfigParser()
config.read('.user.ini')
connection = dbapi.connect(
    address=config.get('hana', 'url'), 
    port=config.get('hana', 'port'), 
    user=config.get('hana', 'user'),
    password=config.get('hana', 'passwd'),
    autocommit=True,
    sslValidateCertificate=False
)

EMBEDDING_DEPLOYMENT_ID = "df09b8224283acd3"
LLM_DEPLOYMENT_ID = "d1507b82d2608a31"

# Define which model to use
chat_llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)

# Load custom documents
loader = TextLoader('./data/information')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
print(f"Number of document chunks: {len(texts)}")

# Create embeddings for custom documents
embeddings = OpenAIEmbeddings(deployment_id=EMBEDDING_DEPLOYMENT_ID)
db = HanaDB(
    embedding=embeddings, connection=connection, table_name="EMBEDDINGS_SHAWKING"
)

# Delete already existing documents from the table
db.delete(filter={})

# add the loaded document chunks
db.add_documents(texts)

# Create a retriever instance of the vector store
retriever = db.as_retriever()

# Create the QA instance to query llm based on custom documents
qa = RetrievalQA.from_llm(llm=chat_llm, retriever=retriever)

# Send query
query = "Wann hat Robert Keck Geburtstag ?"

os.system('cls||clear')
print("\n\n\nRESPONSE:")
print(qa.invoke(query))
print("\n\n\n")