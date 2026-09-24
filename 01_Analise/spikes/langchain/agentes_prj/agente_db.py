import os
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool, tool
from langchain.agents import create_agent

# Carregar variáveis de ambiente
load_dotenv()
emb_model = os.getenv("EMB_MODEL")
ollama_model = os.getenv("OLLAMA_MODEL")

# Configurar embeddings com Ollama local
embeddings = OllamaEmbeddings(
    model=emb_model,
    base_url="http://localhost:11434"
)

# Carregar dados do JSON
def load_person_data(json_file):
    """Carrega dados de pessoas de um arquivo JSON"""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Carregar json
json_file = "01_Analise/spikes/langchain/agentes_prj/db.json"
person_data = load_person_data(json_file)


def _format_person(person: dict) -> str:
    return (
        f"Nome: {person.get('nome', 'N/A')}\n"
        f"Idade: {person.get('idade', 'N/A')}\n"
        f"Cartão Cidadão: {person.get('cartao_cidadao', 'N/A')}\n"
        f"NIF: {person.get('nif', 'N/A')}\n"
        f"Data Nascimento: {person.get('data_nascimento', 'N/A')}\n"
        f"Profissão: {person.get('profissao', 'N/A')}\n"
        f"Cidade: {person.get('cidade', 'N/A')}\n"
        f"Distrito: {person.get('distrito', 'N/A')}\n"
        f"Morada: {person.get('morada', 'N/A')}\n"
        f"Código Postal: {person.get('codigo_postal', 'N/A')}\n"
        f"Telefone: {person.get('telefone', 'N/A')}\n"
        f"Email: {person.get('email', 'N/A')}\n"
        f"Estado Civil: {person.get('estado_civil', 'N/A')}\n"
        f"Nacionalidade: {person.get('nacionalidade', 'N/A')}"
    )


@tool
def list_all_people() -> str:
    """Retorna a lista COMPLETA de todas as pessoas no banco de dados,
    sem nenhum filtro. Usa esta ferramenta sempre que for pedido
    'todas as pessoas' ou uma listagem geral."""
    return "\n\n---\n\n".join(_format_person(p) for p in person_data)


@tool
def get_person_by_document(numero: str) -> str:
    """Procura uma pessoa pelo número EXATO de cartão de cidadão (CC) ou NIF.
    Usa esta ferramenta para qualquer pergunta que mencione um número de
    CC ou NIF. Argumento: o número tal como foi escrito na pergunta."""
    numero = numero.strip()
    matches = [
        p for p in person_data
        if p.get("cartao_cidadao") == numero or p.get("nif") == numero
    ]
    if not matches:
        return f"Nenhuma pessoa encontrada com o número '{numero}'."
    return "\n\n---\n\n".join(_format_person(p) for p in matches)


@tool
def get_person_by_contact(valor: str) -> str:
    """Procura uma pessoa pelo número de telefone EXATO ou endereço de
    email EXATO. Argumento: o telefone ou email tal como foi escrito
    na pergunta."""
    valor = valor.strip().lower()
    matches = [
        p for p in person_data
        if p.get("telefone", "").lower() == valor
        or p.get("email", "").lower() == valor
    ]
    if not matches:
        return f"Nenhuma pessoa encontrada com o contacto '{valor}'."
    return "\n\n---\n\n".join(_format_person(p) for p in matches)


@tool
def filter_by_field(campo: str, valor: str) -> str:
    """Filtra pessoas por um campo EXATO específico, como cidade, distrito,
    profissao, estado_civil ou idade. Argumentos: campo (nome do campo,
    ex: 'cidade') e valor (o valor exato a procurar, ex: 'Porto')."""
    campo = campo.strip().lower()
    valor_norm = valor.strip().lower()
    matches = [
        p for p in person_data
        if str(p.get(campo, "")).strip().lower() == valor_norm
    ]
    if not matches:
        return f"Nenhuma pessoa encontrada com {campo} = '{valor}'."
    return "\n\n---\n\n".join(_format_person(p) for p in matches)

# Converter dados para textos para indexação - MELHORADO
texts = []
for person in person_data:
    # Criar uma representação textual mais rica e com campos específicos
    person_text = f"""
    Nome: {person.get('nome', 'N/A')}
    Idade: {person.get('idade', 'N/A')}
    Cartão Cidadão: {person.get('cartao_cidadao', 'N/A')}
    NIF: {person.get('nif', 'N/A')}
    Data Nascimento: {person.get('data_nascimento', 'N/A')}
    Profissão: {person.get('profissao', 'N/A')}
    Cidade: {person.get('cidade', 'N/A')}
    Distrito: {person.get('distrito', 'N/A')}
    Morada: {person.get('morada', 'N/A')}
    Código Postal: {person.get('codigo_postal', 'N/A')}
    Telefone: {person.get('telefone', 'N/A')}
    Email: {person.get('email', 'N/A')}
    Estado Civil: {person.get('estado_civil', 'N/A')}
    Nacionalidade: {person.get('nacionalidade', 'N/A')}
    """
    texts.append(person_text)

# Criar vector store com os dados das pessoas
vector_store = FAISS.from_texts(texts, embedding=embeddings)

# Testar procura
print("Testando procura:")
print(vector_store.similarity_search('João Silva', k=2))

# Criar retriever com mais resultados
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Aumentado para 5

retriever_tool = create_retriever_tool(
    retriever,
    name="person_search_fuzzy",
    description=(
        "procura APROXIMADA/semântica no banco de dados de pessoas. Usa isto "
        "APENAS quando não souberes o nome exato, ou para perguntas vagas "
        "tipo 'alguém que trabalhe com engenharia'. NÃO uses isto para "
        "números de CC/NIF/telefone, nem para listar todas as pessoas - "
        "não é fiável para isso porque pode ignorar resultados relevantes."
    )
)

# Criar agente com sistema prompt atualizado e CORRIGIDO
agent = create_agent(
    model=f"ollama:{ollama_model}",
    tools=[
        list_all_people,
        get_person_by_document,
        get_person_by_contact,
        filter_by_field,
        retriever_tool,
    ],
    system_prompt = (
        "És um assistente especializado num banco de dados de pessoas em Portugal. "
        "Tens VÁRIAS ferramentas disponíveis, cada uma para um tipo diferente de pergunta. "
        "Escolhe SEMPRE a ferramenta exata mais adequada antes de recorrer à procura fuzzy:\n\n"

        "• list_all_people: quando for pedido 'todas as pessoas' ou uma listagem geral.\n"
        "• get_person_by_document: quando a pergunta mencionar um número de cartão de "
        "cidadão (CC) ou NIF.\n"
        "• get_person_by_contact: quando a pergunta mencionar um telefone ou email.\n"
        "• filter_by_field: quando a pergunta pedir pessoas por cidade, distrito, "
        "profissão, estado civil ou idade EXATOS (ex: campo='cidade', valor='Porto').\n"
        "• person_search_fuzzy: SÓ como último recurso, para nomes parciais ou "
        "perguntas vagas onde nenhuma das ferramentas acima se aplica.\n\n"

        "ESTRUTURA DO BANCO DE DADOS:\n"
        "nome, idade, cartao_cidadao, nif, data_nascimento, profissao, cidade, distrito, "
        "morada, codigo_postal, telefone, email, estado_civil, nacionalidade\n\n"

        "REGRAS CRÍTICAS:\n"
        "• Pesquisa SEMPRE pelo nome com o seu primeiro e último (ex: António Luís dos Santos fica António Santos)\n"
        "• NUNCA inventes informações - usa apenas o que as ferramentas retornarem\n"
        "• Verifica SEMPRE se os dados correspondem exatamente ao que foi perguntado\n"
        "• Se um campo mostra 'N/A', o dado não está disponível\n"
        "• Lista TODAS as pessoas relevantes quando houver múltiplos resultados\n"
        "• Se uma ferramenta não encontrar nada, diz isso claramente em vez de inventar\n"
    )
)

# Função para fazer perguntas sobre pessoas
def ask_about_people(question):
    """Faz uma pergunta sobre as pessoas no banco de dados"""
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content

# Exemplos de perguntas que podem ser feitas
print("\n=== Exemplos de perguntas ===")

# Pergunta 1: procurar informações específicas - CORRIGIDO
#result1 = ask_about_people("Qual é a idade do João Santos?")
result1 = ask_about_people("Qual é a idade do João Miguel Silva Santos?")
print(f"Pergunta: Qual é a idade do João Miguel Silva Santos")
print(f"Resposta: {result1}\n")

# Pergunta 2: Listar pessoas com critérios - CORRIGIDO
result2 = ask_about_people("Quem são as pessoas que moram no Porto?")
print(f"Pergunta: Quem são as pessoas que moram no Porto?")
print(f"Resposta: {result2}\n")

# Pergunta 3: procurar por múltiplas informações - CORRIGIDO
result3 = ask_about_people("Mostre-me o nome, idade e cidade de todas as pessoas")
print(f"Pergunta: Mostre-me o nome, idade e cidade de todas as pessoas")
print(f"Resposta: {result3}\n")

# Pergunta 4: Consulta específica - CORRIGIDO
result4 = ask_about_people("Quem tem o cartão de cidadão 987654321?")
print(f"Pergunta: Quem tem o cartão de cidadão 987654321?")
print(f"Resposta: {result4}\n")