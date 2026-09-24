import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
import unicodedata

# Para documentos ODT
from odf.opendocument import OpenDocumentText
from odf.text import P, H
from odf.style import Style, TextProperties

# Carregar variáveis de ambiente
load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")

# Dados das pessoas
def load_person_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

json_file = "01_Analise/spikes/langchain/agentes_prj/db.json"
person_data = load_person_data(json_file)

def normalizar_texto(texto):
    """Remove acentos e caracteres especiais para facilitar procura"""
    if not texto:
        return ""
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return texto

class DocumentGenerator:
    """Classe simples para gerar documentos ODT"""
    
    @staticmethod
    def criar_documento_odt(titulo, conteudo, nome_arquivo=None):
        """Cria um documento ODT com título e conteúdo"""
        doc = OpenDocumentText()
        
        # Estilo para título
        titulo_estilo = Style(name="TituloEstilo", family="paragraph")
        titulo_props = TextProperties(fontsize="16pt", fontweight="bold")
        titulo_estilo.addElement(titulo_props)
        doc.styles.addElement(titulo_estilo)
        
        # Estilo para corpo
        corpo_estilo = Style(name="CorpoEstilo", family="paragraph")
        corpo_props = TextProperties(fontsize="11pt")
        corpo_estilo.addElement(corpo_props)
        doc.styles.addElement(corpo_estilo)
        
        # Adicionar título
        h = H(outlinelevel=1, stylename=titulo_estilo, text=titulo)
        doc.text.addElement(h)
        doc.text.addElement(P(text=""))
        
        # Adicionar conteúdo
        for paragraph in conteudo:
            if paragraph.strip():
                p = P(stylename=corpo_estilo, text=paragraph)
                doc.text.addElement(p)
        
        # Salvar
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"documento_{timestamp}.odt"
        
        doc.save(nome_arquivo)
        return nome_arquivo

# Ferramentas
@tool
def procurar_pessoa(nome: str) -> str:
    """procura uma pessoa pelo nome e retorna seus dados"""
    if not nome:
        return "Por favor, forneça um nome para procurar."
    
    nome_procura = normalizar_texto(nome)
    
    matches = []
    for person in person_data:
        nome_pessoa = person.get('nome', '')
        nome_normalizado = normalizar_texto(nome_pessoa)
        
        if (nome_procura in nome_normalizado or 
            nome_normalizado in nome_procura or
            any(palavra in nome_normalizado for palavra in nome_procura.split())):
            matches.append(person)
    
    if not matches:
        nomes_disponiveis = "\n".join([f"• {p['nome']}" for p in person_data[:10]])
        if len(person_data) > 10:
            nomes_disponiveis += f"\n... e mais {len(person_data) - 10} pessoas"
        
        return (f"Nenhuma pessoa encontrada com o nome '{nome}'.\n\n"
                f"Pessoas disponíveis no banco de dados:\n{nomes_disponiveis}")
    
    if len(matches) > 1:
        nomes = "\n".join([f"• {p['nome']}" for p in matches])
        return (f"Encontrei várias pessoas com nomes semelhantes a '{nome}':\n"
                f"{nomes}\n\n"
                f"Por favor, especifique o nome completo ou use um dos nomes acima.")
    
    person = matches[0]
    dados = []
    for key, value in person.items():
        dados.append(f"{key.capitalize()}: {value}")
    
    return "\n".join(dados)

@tool
def criar_documento_simples(titulo: str, conteudo: str) -> str:
    """Cria um documento ODT com título e conteúdo fornecidos"""
    if not titulo or not conteudo:
        return "É necessário fornecer título e conteúdo."
    
    paragrafos = [p.strip() for p in conteudo.split('\n') if p.strip()]
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"documento_{timestamp}.odt"
        
        DocumentGenerator.criar_documento_odt(titulo, paragrafos, nome_arquivo)
        return f"Documento criado com sucesso: {nome_arquivo}"
    except Exception as e:
        return f"Erro ao criar documento: {str(e)}"

@tool
def criar_documento_pessoa(nome_pessoa: str, titulo: str = None) -> str:
    """Cria um documento com os dados de uma pessoa específica"""
    if not nome_pessoa:
        return "Por favor, forneça o nome da pessoa."
    
    pessoa = None
    nome_procura = normalizar_texto(nome_pessoa)
    
    for p in person_data:
        nome_normalizado = normalizar_texto(p.get('nome', ''))
        if (nome_procura in nome_normalizado or 
            nome_normalizado in nome_procura or
            any(palavra in nome_normalizado for palavra in nome_procura.split())):
            pessoa = p
            break
    
    if not pessoa:
        nomes_disponiveis = "\n".join([f"• {p['nome']}" for p in person_data[:10]])
        if len(person_data) > 10:
            nomes_disponiveis += f"\n... e mais {len(person_data) - 10} pessoas"
        
        return (f"Pessoa '{nome_pessoa}' não encontrada.\n\n"
                f"Pessoas disponíveis:\n{nomes_disponiveis}\n\n"
                f"Por favor, use o nome completo ou tente um nome da lista.")
    
    if titulo is None:
        titulo = f"Dados de {pessoa['nome']}"
    
    conteudo = [
        f"Data: {datetime.now().strftime('%d/%m/%Y')}",
        "",
        "=== DADOS PESSOAIS ===",
        f"Nome: {pessoa.get('nome', 'N/A')}",
        f"Idade: {pessoa.get('idade', 'N/A')}",
        f"Data Nascimento: {pessoa.get('data_nascimento', 'N/A')}",
        f"NIF: {pessoa.get('nif', 'N/A')}",
        f"Cartão Cidadão: {pessoa.get('cartao_cidadao', 'N/A')}",
        "",
        "=== INFORMAÇÕES PROFISSIONAIS ===",
        f"Profissão: {pessoa.get('profissao', 'N/A')}",
        "",
        "=== CONTACTO E MORADA ===",
        f"Cidade: {pessoa.get('cidade', 'N/A')}",
        f"Distrito: {pessoa.get('distrito', 'N/A')}",
        f"Morada: {pessoa.get('morada', 'N/A')}",
        f"Código Postal: {pessoa.get('codigo_postal', 'N/A')}",
        f"Telefone: {pessoa.get('telefone', 'N/A')}",
        f"Email: {pessoa.get('email', 'N/A')}",
        "",
        "=== OUTRAS INFORMAÇÕES ===",
        f"Estado Civil: {pessoa.get('estado_civil', 'N/A')}",
        f"Nacionalidade: {pessoa.get('nacionalidade', 'N/A')}"
    ]
    
    try:
        nome_arquivo = f"{pessoa['nome'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.odt"
        DocumentGenerator.criar_documento_odt(titulo, conteudo, nome_arquivo)
        return f"Documento criado para {pessoa['nome']}: {nome_arquivo}"
    except Exception as e:
        return f"Erro ao criar documento: {str(e)}"

@tool
def criar_documento_por_filtro(campo: str, valor: str, titulo: str = None) -> str:
    """
    Filtra as pessoas por um determinado campo (ex: cidade, distrito, profissao) 
    e gera um documento ODT com a lista completa dessas pessoas.
    """
    if not campo or not valor:
        return "Por favor, forneça o campo e o valor para o filtro."
    
    campo_norm = campo.strip().lower()
    valor_norm = normalizar_texto(valor)
    
    # Filtrar pessoas
    correspondencias = []
    for p in person_data:
        val_campo = normalizar_texto(str(p.get(campo_norm, '')))
        if valor_norm in val_campo:
            correspondencias.append(p)
            
    if not correspondencias:
        return f"Nenhuma pessoa encontrada com {campo} = '{valor}'."
    
    if titulo is None:
        titulo = f"Relatório de Pessoas - {campo.capitalize()}: {valor.capitalize()}"
        
    conteudo = [
        f"Data de Emissão: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Filtro Aplicado: {campo.capitalize()} = {valor}",
        f"Total de Registos Encontrados: {len(correspondencias)}",
        "--------------------------------------------------"
    ]
    
    for i, p in enumerate(correspondencias, 1):
        conteudo.extend([
            f"\nREGISTO #{i}",
            f"Nome: {p.get('nome', 'N/A')}",
            f"Idade: {p.get('idade', 'N/A')}",
            f"Profissão: {p.get('profissao', 'N/A')}",
            f"Cartão de Cidadão / NIF: {p.get('cartao_cidadao', 'N/A')}",
            f"Morada: {p.get('morada', 'N/A')}, {p.get('codigo_postal', 'N/A')}",
            f"Cidade/Distrito: {p.get('cidade', 'N/A')} / {p.get('distrito', 'N/A')}",
            f"Telefone: {p.get('telefone', 'N/A')}",
            f"Email: {p.get('email', 'N/A')}",
            f"Estado Civil: {p.get('estado_civil', 'N/A')}",
            "--------------------------------------------------"
        ])
        
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"relatorio_{campo_norm}_{valor_norm}_{timestamp}.odt"
        DocumentGenerator.criar_documento_odt(titulo, conteudo, nome_arquivo)
        return f"Documento criado com sucesso com {len(correspondencias)} pessoa(s): {nome_arquivo}"
    except Exception as e:
        return f"Erro ao criar documento: {str(e)}"


# Criar agente
agent = create_agent(
    model=f"ollama:{ollama_model}",
    tools=[
        procurar_pessoa,
        criar_documento_simples,
        criar_documento_pessoa,
        criar_documento_por_filtro,
    ],
    system_prompt=(
        "És um assistente especializado em criar documentos ODT (OpenDocument Text).\n\n"
        "Tens as seguintes ferramentas disponíveis:\n"
        "1. procurar_pessoa: Procura os dados de uma pessoa específica pelo nome.\n"
        "2. criar_documento_simples: Cria um documento quando o utilizador fornece diretamente o título e texto.\n"
        "3. criar_documento_pessoa: Cria um documento individual focado apenas numa pessoa específica.\n"
        "4. criar_documento_por_filtro: Cria um documento com TODAS as pessoas que correspondem a um determinado critério "
        "(ex: cidade='Porto', distrito='Lisboa', profissao='Médica').\n\n"
        
        "REGRAS IMPORTANTES:\n"
        "• Para pedidos como 'gera um documento de todas as pessoas que vivem no Porto', usa SEMPRE 'criar_documento_por_filtro' com campo='cidade' e valor='Porto'.\n"
        "• Para pedidos sobre uma pessoa em concreto, usa 'criar_documento_pessoa'.\n"
        "• NUNCA inventes dados. Usa exclusivamente as ferramentas fornecidas para interagir com o banco de dados.\n"
        "• Responde sempre em português de Portugal indicando o nome do ficheiro .odt criado."
    )
)

def ask_document(question):
    """Faz uma pergunta ao agente de geração de documentos"""
    print(f"\nPergunta: {question}")
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    resposta = result["messages"][-1].content
    print(f"Resposta:\n{resposta}\n")
    return resposta

if __name__ == "__main__":
    print("=== GERADOR DE DOCUMENTOS ODT ===\n")
    
    # Teste para a tua task específica:
    ask_document("Gera um documento com a informação de todas as pessoas que vivem no Porto")