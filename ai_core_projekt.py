from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="cred.env")  # Lädt die Umgebungsvariablen aus der .env-Datei

print("AICORE_BASE_URL:", os.getenv("AICORE_BASE_URL"))
print("AICORE_AUTH_URL:", os.getenv("AICORE_AUTH_URL"))
print("AICORE_CLIENT_ID:", os.getenv("AICORE_CLIENT_ID"))
print("AICORE_CLIENT_SECRET:", os.getenv("AICORE_CLIENT_SECRET"))
print("AICORE_RESOURCE_GROUP:", os.getenv("AICORE_RESOURCE_GROUP"))


from gen_ai_hub.proxy.native.openai import chat

messages = [{"role": "system", "content": "Du bist ein hilfsbereiter Assistent."},
            {"role": "user", "content": "Wie viel wiegt eine Giraffe?"}]

kwargs = dict(model_name='gpt-4o-mini', messages=messages)
response = chat.completions.create(**kwargs)

print(response)



