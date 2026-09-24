import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

# Carregar variáveis de ambiente (se necessário)
load_dotenv()
emb_model = os.getenv("EMB_MODEL")
ollama_model = os.getenv("OLLAMA_MODEL")

# Configurar embeddings com Ollama local
# Certifique-se de que o Ollama está rodando e o modelo está disponível
embeddings = OllamaEmbeddings(
    model = emb_model,
    base_url="http://localhost:11434"
)

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('What fruits does the person like?', k=3))
print(vector_store.similarity_search('What fruits does the person hate?', k=3))

retriever = vector_store.as_retriever(search_kwargs={"k": 6})

retriever_tool = create_retriever_tool(
    retriever, 
    name="kb_search", 
    description="Search the small product / fruit database for information.")

agent = create_agent(
    model=f"ollama:{ollama_model}",
    tools=[retriever_tool],
    system_prompt = (
    "You are a helpful assistant with access to a kb_search tool over a small "
    "database of statements about fruits and other things the person likes or dislikes. "
    "When asked about what the person likes and dislikes, you MUST call kb_search "
    "at least twice: once with a query focused on things liked/enjoyed, and once "
    "with a query focused on things disliked/hated. Only answer after you have "
    "results covering both categories. Answer succinctly based only on retrieved context.")
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result)
print(result["messages"][-1].content)