from dotenv import load_dotenv

load_dotenv(dotenv_path="cred.env")  # Lädt die Umgebungsvariablen aus der .env-Datei
from gen_ai_hub.proxy.native.openai import embeddings

response = embeddings.create(
    input = "dog",
    model_name="text-embedding-3-small"
)
print(response.data)