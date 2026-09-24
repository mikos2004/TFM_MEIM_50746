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
from langchain.tools import ToolRuntime, tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from dataclasses import dataclass


load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")


@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool(
    "get_weather",
    description="Returns weather information for a given city.",
    return_direct=False
)
@tool(
    "get_weather",
    description="Returns current weather information for a given city.",
    return_direct=False
)
def get_weather(city: str):
    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=10,
            headers={"User-Agent": "curl/8.0"}  # wttr.in trata melhor pedidos com este header
        )
        response.raise_for_status()

        if not response.text.strip():
            return {"error": "Resposta vazia da API meteorológica"}

        data = response.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        return {
            "city": area["areaName"][0]["value"],
            "country": area["country"][0]["value"],
            "temp_C": current["temp_C"],
            "temp_F": current["temp_F"],
            "feels_like_C": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "description": current["weatherDesc"][0]["value"],
            "wind_speed_kmph": current["windspeedKmph"],
        }

    except requests.exceptions.Timeout:
        return {"error": "Tempo limite excedido ao contactar a API meteorológica"}
    except requests.exceptions.ConnectionError:
        return {"error": "Erro de conexão com a API meteorológica"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Erro HTTP: {e.response.status_code} - {e.response.reason}"}
    except requests.exceptions.JSONDecodeError:
        return {"error": "A API meteorológica devolveu uma resposta inválida (não-JSON). Pode estar em causa rate limiting."}
    except (KeyError, IndexError):
        return {"error": "Formato de resposta inesperado da API meteorológica"}
    except Exception as e:
        return {"error": f"Erro inesperado: {str(e)}"}


@tool(
    "locate_user",
    description="Look up a user's city based on the context"
)
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case "HJKL111":
            return "Paris"
        case _:
            return "Unknown"
        
model = init_chat_model(
    model=ollama_model,
    model_provider="ollama",
    temperature=0.3,
    base_url="http://localhost:11434" 
)

checkpointer = InMemorySaver()


agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="""
    IMPORTANTE: DEVES responder em português (PT / PT-PT) a todas as perguntas dos utilizadores.

    REGRAS OBRIGATÓRIAS DE EXECUÇÃO (segue esta ordem, sem exceções):
    1. Chama SEMPRE a tool locate_user primeiro para descobrir a cidade do utilizador.
    NUNCA perguntes a cidade diretamente ao utilizador — usa sempre locate_user.
    2. Com a cidade obtida, chama SEMPRE a tool get_weather para obter os dados meteorológicos.
    3. Só depois de teres os dados reais de get_weather, produz o ResponseFormat final,
    preenchendo summary, temperature_celsius, temperature_fahrenheit e humidity com
    os valores reais devolvidos pela tool. Nunca inventes ou deixes valores nulos.
    """,
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)



#############################
# Main
#############################
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Como está o tempo? "}]}, 
            config=config, 
            context=Context(user_id="XYZ456")
        )

        structured = result.get("structured_response")
        if structured:
            print(structured)
            print(structured.summary)
            print(structured.temperature_celsius)
        else:
            # O modelo respondeu apenas em texto, sem produzir o formato estruturado
            last_message = result["messages"][-1]
            print("Resposta em texto (sem structured_response):")
            print(last_message.content)

    except Exception as e:
        print(f"Erro ao executar o agente: {e}")