"""
Agente que responde às perguntas em modo stream
"""
#############################
# Bibliotecas
#############################
import os
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()
ollama_model=os.getenv("OLLAMA_MODEL")

model = init_chat_model(
    model = ollama_model,
    model_provider="ollama",
    temperature = 0.1,
    base_url="http://localhost:11434" 
)

# Sem stream

#response = model.invoke("Olá, o que é o Python?")
#print(response.content)

# Com stream

for chunk in model.stream("Olá, o que é o Python?"):
    print(chunk.text, end='', flush=True)

