"""
RAG sobre una CARPETA DE PDFs (Ollama + ChromaDB persistente).

Lee todos los PDFs de una carpeta, extrae el texto, lo parte en chunks, los
embebe (nomic-embed-text) y los indexa en una base vectorial local (ChromaDB
persistente en disco). Despues se pueden hacer preguntas: recupera los chunks
mas relevantes y genera la respuesta con Gemma, citando de que archivo/pagina salio.

El indexado es INCREMENTAL: si un PDF ya fue indexado, se saltea. Con --reindex
se borra el indice y se reconstruye desde cero.

Requisitos:
    - Ollama corriendo con:  ollama pull gemma3 ; ollama pull nomic-embed-text
    - pip install -r requirements-pdf.txt

Uso:
    # 1) Poné tus PDFs en la carpeta docs/ (o pasá otra con --docs)
    # 2) Indexar + preguntar:
    python rag_pdf.py "¿De que trata el documento X?"
    python rag_pdf.py                         # interactivo
    python rag_pdf.py --docs C:\ruta\a\pdfs   # otra carpeta
    python rag_pdf.py --reindex "..."         # reconstruye el indice
"""
from __future__ import annotations

import os
import sys
import shutil

import chromadb
import ollama
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_pdf")
COLLECTION_NAME = "pdf_docs"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_K = 4
CHUNK_SIZE = 800      # caracteres por chunk
CHUNK_OVERLAP = 150   # solapamiento entre chunks


def embed(texto: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]


def chunk_text(texto: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Parte un texto en fragmentos con solapamiento."""
    texto = " ".join(texto.split())  # normaliza espacios/saltos
    if not texto:
        return []
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + size
        chunks.append(texto[inicio:fin])
        inicio += size - overlap
    return chunks


def extraer_chunks_pdf(ruta: str) -> list[dict]:
    """Extrae el texto de un PDF y devuelve una lista de chunks con metadata."""
    reader = PdfReader(ruta)
    nombre = os.path.basename(ruta)
    items = []
    for n_pagina, page in enumerate(reader.pages, start=1):
        texto = page.extract_text() or ""
        for i, ch in enumerate(chunk_text(texto)):
            items.append({
                "id": f"{nombre}::p{n_pagina}::c{i}",
                "text": ch,
                "metadata": {"source": nombre, "page": n_pagina},
            })
    return items


def get_collection(reindex: bool = False):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def pdf_ya_indexado(collection, nombre: str) -> bool:
    res = collection.get(where={"source": nombre}, limit=1)
    return len(res.get("ids", [])) > 0


def indexar(collection, docs_dir: str) -> None:
    pdfs = [f for f in sorted(os.listdir(docs_dir)) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"[aviso] No hay PDFs en {docs_dir}. Poné tus archivos ahi y volvé a correr.")
        return

    total_nuevos = 0
    for nombre in pdfs:
        if pdf_ya_indexado(collection, nombre):
            print(f"  = {nombre} (ya indexado, se saltea)")
            continue
        ruta = os.path.join(docs_dir, nombre)
        try:
            items = extraer_chunks_pdf(ruta)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {nombre}: no se pudo leer ({e})")
            continue
        if not items:
            print(f"  ! {nombre}: sin texto extraible (¿PDF escaneado?)")
            continue
        collection.add(
            ids=[it["id"] for it in items],
            embeddings=[embed(it["text"]) for it in items],
            documents=[it["text"] for it in items],
            metadatas=[it["metadata"] for it in items],
        )
        total_nuevos += len(items)
        print(f"  + {nombre}: {len(items)} chunks indexados")

    print(f"\nIndexado listo. Chunks nuevos: {total_nuevos}. "
          f"Total en la base: {collection.count()}.\n")


def recuperar(collection, pregunta: str, k: int = TOP_K) -> list[dict]:
    q = embed(pregunta)
    res = collection.query(query_embeddings=[q], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]


def generar_respuesta(pregunta: str, contexto: list[dict]) -> str:
    bloques = []
    for c in contexto:
        fuente = f"{c['meta'].get('source')} (pag. {c['meta'].get('page')})"
        bloques.append(f"[{fuente}]\n{c['text']}")
    contexto_str = "\n\n".join(bloques)
    prompt = f"""Respondé la pregunta usando SOLO la informacion del contexto extraido
de los PDFs. Si el contexto no alcanza, decilo. Citá las fuentes entre corchetes.

Contexto:
{contexto_str}

Pregunta: {pregunta}

Respuesta:"""
    resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def consultar(collection, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    contexto = recuperar(collection, pregunta)
    if not contexto:
        print("No hay documentos indexados todavia.")
        return
    print("\n--- Fragmentos recuperados ---")
    for c in contexto:
        fuente = f"{c['meta'].get('source')} p.{c['meta'].get('page')}"
        print(f"  * [{fuente}] {c['text'][:80]}...")
    print("\n--- Respuesta del modelo ---")
    print(generar_respuesta(pregunta, contexto))


def main() -> None:
    argv = sys.argv[1:]
    reindex = "--reindex" in argv
    argv = [a for a in argv if a != "--reindex"]

    docs_dir = DOCS_DIR
    if "--docs" in argv:
        i = argv.index("--docs")
        docs_dir = argv[i + 1]
        del argv[i:i + 2]

    os.makedirs(docs_dir, exist_ok=True)
    collection = get_collection(reindex=reindex)

    print(f"Carpeta de PDFs: {docs_dir}")
    print("Indexando PDFs (incremental)...")
    indexar(collection, docs_dir)

    if collection.count() == 0:
        print("No hay nada indexado. Agregá PDFs y volvé a correr.")
        return

    if argv:
        consultar(collection, " ".join(argv))
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
                consultar(collection, pregunta)


if __name__ == "__main__":
    main()
