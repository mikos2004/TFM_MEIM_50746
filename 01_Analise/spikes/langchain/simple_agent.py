"""
Agente simples que dá informações sobre o tempo
"""
#############################
# Bibliotecas
#############################
import os
import requests
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool


load_dotenv()
ollama_model=os.getenv("OLLAMA_MODEL")


@tool(
    "get_weather",
    description="Returns weather information for a given city.",
    return_direct=False
)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


llm = ChatOllama(
    model=ollama_model,
    temperature=0.7,
    base_url="http://localhost:11434",
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=
    """
    IMPORTANTE: DEVES responder em português a todas as perguntas dos utilizadores.
    Mesmo que os dados meteorológicos estejam em inglês, traduz a resposta para português.
    Utiliza o português em todas as suas respostas.
    És um assistente prestável que fornece informações meteorológicas.
    """
)



#############################
# Main
#############################
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Qual o tempo em Lisboa? "}]}
    )
    print(result["messages"][-1].content)
    #print(result)