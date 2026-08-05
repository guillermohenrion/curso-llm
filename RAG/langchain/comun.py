"""
Utilidades compartidas por los ejemplos de RAG con LangChain + Ollama.

Todos los ejemplos usan los mismos modelos locales (via Ollama) y el mismo
corpus de juguete, para que la diferencia entre versiones sea solo la tecnica
de RAG (basico -> memoria -> tools -> skills) y no los datos.

Requisitos:
    - Ollama corriendo (https://ollama.com)
    - Modelos descargados:
        ollama pull gemma3
        ollama pull nomic-embed-text
    - Dependencias:  pip install -r requirements-langchain.txt
"""
from __future__ import annotations

import sys

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# UTF-8 en la salida (util en Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHAT_MODEL = "gemma3"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "clase_rag_langchain"

# --- Corpus de ejemplo (los "documentos de la clase") ---
DOCUMENTOS = [
    {
        "id": "doc1",
        "text": (
            "RAG (Retrieval-Augmented Generation) es una tecnica que combina la "
            "busqueda de informacion con la generacion de texto: primero se recuperan "
            "documentos relevantes de una base de conocimiento, y luego un modelo de "
            "lenguaje genera la respuesta usando esos documentos como contexto."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Una base de datos vectorial almacena embeddings (vectores numericos que "
            "representan el significado de un texto) y permite buscar los vectores mas "
            "similares a una consulta usando metricas como la distancia coseno o euclidiana."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Ollama es una herramienta que permite correr modelos de lenguaje grandes "
            "de forma local, sin depender de una API en la nube. Soporta modelos como "
            "Gemma, Llama y modelos de embeddings como nomic-embed-text."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "ChromaDB es una base de datos vectorial embebida, pensada para prototipos "
            "y aplicaciones RAG: se puede usar en memoria o persistir en disco, sin "
            "necesidad de levantar un servidor aparte."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "El pipeline de un sistema RAG tipico tiene tres etapas: ingesta y chunking "
            "de documentos, indexado en una base vectorial, y en tiempo de consulta, "
            "recuperacion de los chunks mas relevantes seguida de generacion de la respuesta."
        ),
    },
    {
        "id": "doc6",
        "text": (
            "LangChain es un framework para construir aplicaciones con LLMs. Ofrece "
            "componentes como prompts, cadenas (chains), retrievers, memoria, agentes "
            "y herramientas (tools), y un lenguaje de composicion llamado LCEL."
        ),
    },
]


def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Modelo de chat local (Gemma via Ollama). temperature=0 -> respuestas mas deterministas."""
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


def get_embeddings() -> OllamaEmbeddings:
    """Modelo de embeddings local (nomic-embed-text via Ollama)."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_vectorstore() -> Chroma:
    """Crea un vector store Chroma EN MEMORIA con el corpus de ejemplo.

    Cada documento se convierte en embedding y se guarda. Al ser en memoria,
    se reconstruye cada vez que corre el script (suficiente para la clase).
    """
    # DOCUMENTOS (dicts) -> Document de LangChain, el formato que espera Chroma.from_documents
    docs = [
        Document(page_content=d["text"], metadata={"id": d["id"]})
        for d in DOCUMENTOS
    ]
    # from_documents embebe cada texto (con get_embeddings()) y arma el vector store de una
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_retriever(k: int = 2):
    """Devuelve un retriever que trae los k documentos mas relevantes (similitud coseno)."""
    return build_vectorstore().as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    """Concatena el contenido de los documentos recuperados en un solo string."""
    return "\n\n".join(f"- {d.page_content}" for d in docs)


def leer_pregunta_o_argv(default: str) -> str:
    """Toma la pregunta de los argumentos de linea de comando, o usa un default."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    return default
