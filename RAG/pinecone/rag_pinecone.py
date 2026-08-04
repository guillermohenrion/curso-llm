"""
RAG simple con PINECONE (base vectorial en la nube).

Igual que ../rag_simple.py, pero en vez de ChromaDB (local) usa Pinecone,
una base de datos vectorial administrada en la nube. El token de Pinecone se
lee de un archivo .env (NUNCA se hardcodea en el codigo).

Embeddings y generacion siguen siendo locales via Ollama:
    - nomic-embed-text  -> embeddings
    - gemma3            -> generacion

Requisitos previos:
    1. Cuenta gratis en https://www.pinecone.io y una API key.
    2. Copiar .env.example a .env y completar PINECONE_API_KEY.
    3. Ollama corriendo con los modelos:
           ollama pull gemma3
           ollama pull nomic-embed-text
    4. pip install -r requirements-pinecone.txt

Flujo:
    1. Indexar: cada documento -> embedding (Ollama) -> upsert en Pinecone.
    2. Recuperar: la pregunta -> embedding -> query top_k en Pinecone.
    3. Generar: prompt con contexto recuperado -> Gemma responde.

Uso:
    python rag_pinecone.py
    python rag_pinecone.py "¿Que es una base de datos vectorial?"
"""
from __future__ import annotations

import os
import sys
import time

import ollama
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

sys.stdout.reconfigure(encoding="utf-8")

# --- Configuracion (se lee del entorno / .env) ---
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "curso-llm-rag")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_K = 2
NAMESPACE = "clase"

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
            "similares a una consulta usando metricas como la distancia coseno."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Ollama permite correr modelos de lenguaje de forma local, sin depender de "
            "una API en la nube. Soporta modelos como Gemma y embeddings como nomic-embed-text."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Pinecone es una base de datos vectorial administrada (en la nube): se crea "
            "un indice, se hace 'upsert' de vectores con metadata, y se consultan los mas "
            "similares sin tener que administrar infraestructura propia."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "El pipeline de un sistema RAG tipico tiene tres etapas: ingesta y chunking, "
            "indexado en una base vectorial, y en consulta, recuperacion de los chunks mas "
            "relevantes seguida de la generacion de la respuesta."
        ),
    },
]


def embed(texto: str) -> list[float]:
    """Devuelve el embedding de un texto usando Ollama."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]


def get_index(pc: Pinecone, dim: int):
    """Crea el indice si no existe (con la dimension detectada) y lo devuelve."""
    existentes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existentes:
        print(f"Creando indice '{INDEX_NAME}' (dim={dim}, metric=cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Esperar a que el indice este listo.
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(INDEX_NAME)


def indexar(index) -> None:
    """Embebe cada documento y hace upsert en Pinecone."""
    vectores = []
    for doc in DOCUMENTOS:
        vectores.append({
            "id": doc["id"],
            "values": embed(doc["text"]),
            "metadata": {"text": doc["text"]},
        })
    index.upsert(vectors=vectores, namespace=NAMESPACE)
    print(f"Indexados {len(vectores)} documentos en Pinecone.\n")


def recuperar(index, pregunta: str, k: int = TOP_K) -> list[str]:
    """Busca en Pinecone los k documentos mas parecidos a la pregunta."""
    q = embed(pregunta)
    res = index.query(
        vector=q, top_k=k, include_metadata=True, namespace=NAMESPACE
    )
    return [m["metadata"]["text"] for m in res["matches"]]


def generar_respuesta(pregunta: str, contexto: list[str]) -> str:
    contexto_str = "\n\n".join(f"- {c}" for c in contexto)
    prompt = f"""Respondé la pregunta usando SOLO la informacion del contexto.
Si el contexto no alcanza, decilo.

Contexto:
{contexto_str}

Pregunta: {pregunta}

Respuesta:"""
    resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def rag(index, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    contexto = recuperar(index, pregunta)
    print("\n--- Documentos recuperados ---")
    for c in contexto:
        print(f"  * {c[:90]}...")
    print("\n--- Respuesta del modelo ---")
    print(generar_respuesta(pregunta, contexto))


def main() -> None:
    if not PINECONE_API_KEY:
        print("ERROR: falta PINECONE_API_KEY.")
        print("Copiá .env.example a .env y completá tu API key de Pinecone.")
        sys.exit(1)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Detectamos la dimension del embedding a partir de un texto de prueba.
    dim = len(embed("dimension check"))
    index = get_index(pc, dim)

    print("Indexando documentos en Pinecone...")
    indexar(index)

    if len(sys.argv) > 1:
        rag(index, " ".join(sys.argv[1:]))
    else:
        print("Escribí una pregunta (o 'salir' para terminar):")
        while True:
            try:
                pregunta = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if pregunta:
                rag(index, pregunta)


if __name__ == "__main__":
    main()
