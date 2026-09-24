#############################
# Bibliotecas
#############################
import asyncio
import json
from fastmcp import Client
import requests

# Configuração do Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b" 

def query_ollama(prompt, model=OLLAMA_MODEL):
    """Consulta o Ollama localmente."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Erro ao conectar ao Ollama: {e}"
    except KeyError:
        return "Erro: Resposta inesperada do Ollama"
    except Exception as e:
        return f"Erro: {e}"

async def main():
    # Conecta ao servidor HTTP do Knowledge Base
    client_mcp = Client("http://localhost:8050/mcp")
    
    async with client_mcp:
        # Verifica se o servidor está à escuta
        await client_mcp.ping()
        print(" >>> Servidor Knowledge Base conectado com sucesso!")
        print()

        # Lista todas as ferramentas disponíveis
        tools = await client_mcp.list_tools()
        print(f" >>> Ferramentas disponíveis:")
        for tool in tools:
            print(f" - {tool.name}: {tool.description}")
        print()

        # --- Exemplo 1: Consultar a base de conhecimento diretamente ---
        print(">>> Consultando a base de conhecimento:")
        kb_result = await client_mcp.call_tool("get_knowledge_base", {})
        
        if kb_result.is_error:
            print(f"Erro: {kb_result.content[0].text}")
        else:
            knowledge_base = kb_result.content[0].text
            print(knowledge_base[:500] + "...")
        print()

        # --- Modo de perguntas contínuas com Ollama ---
        print(">>> Modo de perguntas contínuas (digite 'sair' para terminar):")
        print(f"Usando modelo: {OLLAMA_MODEL} (local)")
        print()
        
        while True:
            # Pega a pergunta do usuário
            question = input("Pergunta: ").strip()
            
            if question.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break
                
            if not question:
                continue
            
            # Obtém a KB atualizada
            kb_result = await client_mcp.call_tool("get_knowledge_base", {})
            if kb_result.is_error:
                print(f"Erro ao obter KB: {kb_result.content[0].text}")
                continue
                
            kb_text = kb_result.content[0].text
            
            # Prepara o prompt
            prompt = f"""
            Você é um assistente de RH. Responda à pergunta do usuário baseando-se APENAS na base de conhecimento fornecida.
            Se a resposta não estiver na base, diga "Não encontrei essa informação na base de conhecimento."
            
            BASE DE CONHECIMENTO:
            {kb_text}
            
            PERGUNTA:
            {question}
            
            RESPOSTA:
            """
            
            print("Processando...", end="", flush=True)
            
            try:
                answer = query_ollama(prompt)
                print("\r" + " " * 20 + "\r", end="")  # Limpa a linha
                print(f"Resposta: {answer}\n")
                
            except Exception as e:
                print(f"\rErro ao chamar Ollama: {e}\n")

#############################
# Main
#############################

if __name__ == "__main__":
    asyncio.run(main())