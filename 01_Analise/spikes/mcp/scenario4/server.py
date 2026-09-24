#############################
# Bibliotecas
#############################
import os
import json
from fastmcp import FastMCP

# integração com LLM

mcp = FastMCP(name="Knowledge Base")

# decorator para declarar funções
@mcp.tool()
def get_knowledge_base() -> str:
    """Retrieve the entire knowledge base as a formatted string.

    Returns:
        A formatted string containing all Q&A pairs from the knowledge base.
    """
    try:
        # Obter o caminho para o arquivo kb.json na pasta data
        # __file__ é o caminho do script atual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(current_dir, "data", "kb.json")
        
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        # Formatar a base de conhecimento como uma string
        kb_text = "Here is the retrieved knowledge base:\n\n"

        if isinstance(kb_data, list):
            for i, item in enumerate(kb_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "Unknown question")
                    answer = item.get("answer", "Unknown answer")
                else:
                    question = f"Item {i}"
                    answer = str(item)

                kb_text += f"Q{i}: {question}\n"
                kb_text += f"A{i}: {answer}\n\n"
        else:
            kb_text += f"Knowledge base content: {json.dumps(kb_data, indent=2)}\n\n"

        return kb_text
    except FileNotFoundError:
        return "Error: Knowledge base file not found"
    except json.JSONDecodeError:
        return "Error: Invalid JSON in knowledge base file"
    except Exception as e:
        return f"Error: {str(e)}"

#############################
# Main
#############################

if __name__ == "__main__":
    transport = "http"  # "stdio" ou "http"

    if transport == "stdio":
        mcp.run(transport="stdio")  # STDIO
    elif transport == "http":
        mcp.run(transport="http", host="localhost", port=8050)
    else:
        raise ValueError(f"Unknown transport: {transport}")