import os
import uuid
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
ollama_model = os.getenv("OLLAMA_MODEL")
emb_model = os.getenv("EMB_MODEL")

llm = init_chat_model(
    model = ollama_model,
    model_provider="ollama",
    base_url="http://localhost:11434" 
)

# Base de conhecimento para o RAG
KNOWLEDGE = [
    "NeuralNine is a YouTube channel focused on programming, AI, and software engineering tutorials.",
    "LangGraph is a library for building stateful, multi-agent applications on top of LangChain.",
    "A StateGraph in LangGraph defines nodes and edges that operate on a shared typed state.",
    "Checkpointers Like InMemorySaver let LangGraph persist conversation state across invocations using a thread_id.",
    "RAG (Retrieval-Augmented Generation) combines a retriever over a knowledge base with an LLM to ground answers in source documents."
]

vector_store = InMemoryVectorStore(OllamaEmbeddings(model=emb_model))
vector_store.add_documents(Document(page_content = text) for text in KNOWLEDGE)

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user wants to just chat, ask for knowledge or change code in the project.')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code").'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])
    return {'message_intent': result.message_intent}


def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_rag(state: State):
    query = state['messages'][-1].content
    
    # Estratégia 1: Procura semântica normal
    documents = vector_store.similarity_search(query, k=5)
    
    # Estratégia 2: Procura por palavras-chave (para nomes próprios)
    query_lower = query.lower()
    keyword_matches = []
    for doc in KNOWLEDGE:
        if any(keyword in doc.lower() for keyword in query_lower.split()):
            keyword_matches.append(doc)
    
    # Combinar resultados (priorizar matches por palavra-chave)
    all_docs = keyword_matches + [doc.page_content for doc in documents]
    unique_docs = list(dict.fromkeys(all_docs))[:5]  # Remover duplicatas
    
    context = '\n'.join(f'- {doc}' for doc in unique_docs)
    
    # Debug
    #print(f"DEBUG - Query: {query}")
    #print(f"DEBUG - Context recuperado: {context}")
    
    messages = [
        {'role': 'system', 'content': f"""You are a RAG (Retrieval-Augmented Generation) agent.

CONTEXT (use ONLY this information):
{context}

INSTRUCTIONS:
1. Answer the user's question using ONLY the context above
2. If the context contains the answer, provide it clearly and directly
3. If the context does NOT contain the answer, respond with: "I don't have that information in my knowledge base."
4. The context may contain answers even if the phrasing is different
5. Look for synonyms and related terms in the context

Question: """},
    ] + state['messages']

    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_code(state: State):
    messages = [{'role': 'system', 'content': 'No matter what the user says, always say "I am the CODING agent"'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]}



graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'coding_agent'})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input('Enter message:')
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)
    
    # Filtra a última mensagem do assistente
    last_message = result['messages'][-1]
    if hasattr(last_message, 'content'):
        print(last_message.content)