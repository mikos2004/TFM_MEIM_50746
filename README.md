# An Agentic AI Framework for Assisting Cybercrime Investigations in Simulated Environments

---

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-FF6F61?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/MCP-Model%20Context%20Protocol-6E4AFF?style=for-the-badge" alt="MCP">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLMs-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge" alt="Status">
</p>

---

##  Índice

- [Estrutura do Projeto](#-estrutura-do-projeto)

- [Instalação](#-instalação)
- [Módulos e Componentes](#-módulos-e-componentes)
- [Como Utilizar](#-como-utilizar)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

##  Estrutura do Projeto

```
.
├── 01_Analise/
│   └── spikes/
│       ├── eval/                         # Avaliação de consistência
│       │   ├── resultados/               # Resultados e ficheiro CSV a comparar as diferentes abordagens com WMD
│       │   ├── samples/                  # Textos TXT para análise de consistência
│       │   ├── bwd_txt_TFIDF.py          # Word Mover's Distance com filtragem IDF
│       │   ├── bwd_txt.py                # Word Mover's Distance para textos
│       │   ├── bwd.py                    # Word Mover's Distance para frases predefinidas
│       │   ├── compare_wmd.py            # Compara as abordagens WMD testadas
│       │   └── cos_sim.py                # Cosine Similarity
│       │
│       ├── langchain/                    # Spikes com LangChain
│       │   ├── agente_prj/
│       │   │   ├── agente_db.py          # Agente que lê de uma base de dados
│       │   │   ├── agente_doc.py         # Agente que gera documento a partir da BD
│       │   │   └── db.json               # Base de dados simples para testes
│       │   ├── context_agent.py          # Agente que dá info do tempo com base na cidade
│       │   ├── simple_agent.py           # Agente simples (info do tempo)
│       │   ├── strem_agent.py            # Agente em modo stream
│       │   └── vecdb_agent.py            # Agente com pesquisa em BD vetorial
│       │
│       ├── langgraph/                    # Spikes com LangGraph
│       │   ├── main.py                   # Agente com contexto, BD e classificação de prompt
│       │   └── graph.png                 # Imagem do grafo criado
│       │
│       └── mcp/                          # Spikes com Model Context Protocol
│           ├── scenario1/                # FastMCP (STDIO)
│           ├── scenario2/                # CalcAPI -> MCP (FastAPI + MCP)
│           ├── scenario3/                # Feed Search (FastMCP HTTP)
│           ├── scenario4/                # Client HTTP com base de dados
│           ├── aux_cmd.txt               # Comandos úteis para MCP
│           └── client_http.py            # Cliente HTTP
│
├── install_req.bat                       # Batch para instalar requirements.txt
└── requirements.txt                      # Dependências Python
```

---

##  Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/mikos2004/TFM_MEIM_50746.git
cd TFM_MEIM_50746
```

### 2. Criar e ativar um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

#### Opção A — Via batch script (Windows)

```bash
install_req.bat
```

> **Nota:** Deve editar o nome do ambiente virtual, conforme indicado no ficheiro

#### Opção B — Manual (qualquer sistema)

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente e modelos locais

Cria um ficheiro `.env` na raiz com as chaves e modelos necessáriao:

```env
OLLAMA_MODEL = "qwen3:4b-instruct"
EMB_MODEL = "snowflake-arctic-embed2"
```

Corra o ficheiro `install_local_mod.bat` para instalar os modelos usados do Ollama:

```bash
install_local_mod.bat
```

Em alternativa faça manualmente:

```bash
ollama pull qwen3:4b-instruct
ollama pull snowflake-arctic-embed2
```

---

##  Módulos e Componentes

###  `eval/` — Avaliação de Consistência

Spikes dedicados a medir similaridade e consistência entre textos e embeddings.

| Ficheiro | Descrição |
|---|---|
| `bwd.py` | Word Mover's Distance para frases predefinidas |
| `bwd_txt.py` | Word Mover's Distance para ficheiros de texto |
| `bwd_txt_TFIDF.py` | WMD com filtragem por IDF |
| `cos_sim.py` | Cosine Similarity entre embeddings |
| `samples/` | Textos TXT usados como input |
| `aux_fold/` | Cache de embeddings descarregados |

###  `langchain/` — Agentes com LangChain

| Ficheiro | Descrição |
|---|---|
| `simple_agent.py` | Agente básico com ferramenta de meteorologia |
| `context_agent.py` | Agente contextualizado pela cidade do utilizador |
| `strem_agent.py` | Agente com respostas em modo *stream* |
| `vecdb_agent.py` | Agente com pesquisa em base de dados vetorial |
| `agente_prj/agente_db.py` | Agente que consulta uma BD (`db.json`) |
| `agente_prj/agente_doc.py` | Agente que gera documento com base na BD |

###  `langgraph/` — Orquestração com LangGraph

- `main.py`: agente em grafo que:
  - Responde com base num contexto,
  - Consulta uma base de dados,
  - Classifica o prompt recebido.
- `graph.png`: representação visual do grafo.

###  `mcp/` — Model Context Protocol

Quatro cenários de integração com MCP:

1. **Scenario 1** — FastMCP via STDIO.
2. **Scenario 2** — CalcAPI exposta como MCP (FastAPI + MCP).
3. **Scenario 3** — Feed Search via FastMCP HTTP.
4. **Scenario 4** — Cliente HTTP com conhecimento de uma base de dados.

Ficheiros auxiliares:
- `aux_cmd.txt` — comandos úteis para correr os servidores MCP.
- `client_http.py` — exemplo de cliente HTTP.

---

##  Como Utilizar

> **Nota:** Ao correr algum script que recorra a um modelo **Ollama** deve verificar de que tem o Ollama a correr.
> 
> Verifique que tem os modelos corretos descarregados no Ollama e que estão presentes no ficheiro ``.env``, cujos nomes estão no **ponto 4 da Instalação**.

### Executar um spike de avaliação

```bash
cd 01_Analise/spikes/eval
python bwd_txt.py
```

### Executar um agente LangChain

```bash
cd 01_Analise/spikes/langchain
python simple_agent.py
```

### Executar o grafo LangGraph

```bash
cd 01_Analise/spikes/langgraph
python main.py
```

### Executar um cenário MCP

Exemplo (Scenario 1 — STDIO):

```bash
cd 01_Analise/spikes/mcp/scenario1
python <servidor>.py
```

Consulta `aux_cmd.txt` para comandos específicos de cada cenário.

---