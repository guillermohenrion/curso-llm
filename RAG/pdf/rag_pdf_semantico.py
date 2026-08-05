"""
RAG sobre una CARPETA DE PDFs, con CHUNKING POR ORACIONES (Ollama + ChromaDB persistente).

Es el mismo flujo que rag_pdf.py (extraer texto -> chunkear -> embeber -> indexar ->
recuperar -> generar), pero cambia COMO se arman los chunks:

    rag_pdf.py (chunk_text):
        Corta el texto cada `size` caracteres, con `overlap` de solapamiento.
        Simple y rapido, pero puede partir una palabra u oracion justo al medio
        (ej: "...la base de datos vect" | "orial almacena embeddings...").

    rag_pdf_semantico.py (chunk_por_oraciones, este archivo):
        Primero separa el texto en oraciones, y despues va agrupando oraciones
        COMPLETAS hasta acercarse a un tamano objetivo, sin cortar nunca una
        oracion al medio. Cada chunk es una unidad de sentido mas coherente,
        lo que suele mejorar la calidad de los embeddings y de las citas.

    Trade-off: el tamano de cada chunk es mas variable (una oracion muy larga
    puede hacer que un chunk se pase del objetivo), y el splitter de oraciones
    es una heuristica simple (no un modelo de NLP), asi que en casos raros
    (abreviaturas, numeros con puntos) puede cortar donde no deberia.

Comparte la carpeta docs/ con rag_pdf.py (poné ahi tus PDFs), pero usa su propio
indice persistente (chroma_pdf_semantico/) para no mezclarse con el otro ejemplo.

Requisitos:
    - Ollama corriendo con:  ollama pull gemma3 ; ollama pull nomic-embed-text
    - pip install -r requirements-pdf.txt   (mismas dependencias que rag_pdf.py)

Uso:
    python rag_pdf_semantico.py "¿De que trata el documento X?"
    python rag_pdf_semantico.py                         # interactivo
    python rag_pdf_semantico.py --docs C:\\ruta\\a\\pdfs   # otra carpeta
    python rag_pdf_semantico.py --reindex "..."          # reconstruye el indice
"""
from __future__ import annotations

import os
import re
import sys

import chromadb
import ollama
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_pdf_semantico")
COLLECTION_NAME = "pdf_docs_semantico"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_K = 4
CHUNK_TARGET_SIZE = 800   # caracteres "objetivo" por chunk (igual que rag_pdf.py, para comparar)
CHUNK_MAX_SIZE = 1200     # limite duro: si una oracion sola ya lo supera, queda como chunk propio

# Separador de oraciones simple: corta despues de . ! ? seguido de espacio y mayuscula/fin.
# Es una heuristica (no un modelo de NLP), asi que abreviaturas como "Dr." o "pag. 5"
# ocasionalmente van a generar un corte de oracion donde no corresponde. Para el proposito
# de la demo (comparar contra el chunking por caracteres) alcanza y sobra.
_PATRON_ORACION = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9])")


def embed(texto: str) -> list[float]:
    """Devuelve el embedding de un texto usando Ollama."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]


def separar_oraciones(texto: str) -> list[str]:
    """Parte un texto en oraciones (heuristica simple por puntuacion + mayuscula siguiente)."""
    texto = " ".join(texto.split())  # normaliza espacios/saltos, igual que rag_pdf.py
    if not texto:
        return []
    return [o.strip() for o in _PATRON_ORACION.split(texto) if o.strip()]


def chunk_por_oraciones(
    texto: str, target_size: int = CHUNK_TARGET_SIZE, max_size: int = CHUNK_MAX_SIZE
) -> list[str]:
    """Agrupa oraciones completas en chunks, sin cortar ninguna al medio.

    Va sumando oraciones a un chunk actual hasta que agregar la proxima lo haria
    pasar de `target_size`; ahi cierra el chunk y arranca uno nuevo. Si una sola
    oracion ya es mas larga que `max_size` (caso raro), queda como chunk propio
    en vez de forzar un corte a mitad de oracion.
    """
    oraciones = separar_oraciones(texto)
    if not oraciones:
        return []

    chunks: list[str] = []
    actual: list[str] = []
    largo_actual = 0

    for oracion in oraciones:
        largo_oracion = len(oracion) + 1  # +1 por el espacio que la va a separar de la anterior

        if largo_actual + largo_oracion > target_size and actual:
            # Cerramos el chunk actual (ya tiene contenido) antes de agregar esta oracion.
            chunks.append(" ".join(actual))
            actual = []
            largo_actual = 0

        actual.append(oracion)
        largo_actual += largo_oracion

        # Si una sola oracion ya se paso del maximo, la dejamos sola en su chunk
        # en vez de romperla (preferimos un chunk grande a uno que corte mal).
        if largo_actual > max_size:
            chunks.append(" ".join(actual))
            actual = []
            largo_actual = 0

    if actual:
        chunks.append(" ".join(actual))

    return chunks


def extraer_chunks_pdf(ruta: str) -> list[dict]:
    """Extrae el texto de un PDF y devuelve una lista de chunks (por oraciones) con metadata."""
    reader = PdfReader(ruta)
    nombre = os.path.basename(ruta)
    items = []
    for n_pagina, page in enumerate(reader.pages, start=1):
        texto = page.extract_text() or ""
        for i, ch in enumerate(chunk_por_oraciones(texto)):
            items.append({
                "id": f"{nombre}::p{n_pagina}::c{i}",
                "text": ch,
                "metadata": {"source": nombre, "page": n_pagina},
            })
    return items


def get_collection(reindex: bool = False):
    """Abre (o crea) la coleccion persistente de Chroma. Con reindex=True la borra antes."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def pdf_ya_indexado(collection, nombre: str) -> bool:
    """Chequea si ya hay chunks indexados con source == nombre (indexado incremental)."""
    res = collection.get(where={"source": nombre}, limit=1)
    return len(res.get("ids", [])) > 0


def indexar(collection, docs_dir: str) -> None:
    """Recorre los PDFs de la carpeta e indexa los que todavia no estan en la coleccion."""
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
    """Busca los k chunks mas parecidos a la pregunta, con su metadata (archivo + pagina)."""
    q = embed(pregunta)
    res = collection.query(query_embeddings=[q], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]


def generar_respuesta(pregunta: str, contexto: list[dict]) -> str:
    """Arma el prompt citando la fuente de cada chunk, y le pide a Gemma que responda con eso."""
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
    """Orquesta el flujo completo (recuperar + generar) e imprime los pasos por consola."""
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
    reindex = "--reindex" in argv  # flag: borrar y reconstruir el indice desde cero
    argv = [a for a in argv if a != "--reindex"]

    docs_dir = DOCS_DIR
    if "--docs" in argv:
        # --docs toma el valor siguiente como carpeta; sacamos ambos de argv (flag + valor)
        i = argv.index("--docs")
        docs_dir = argv[i + 1]
        del argv[i:i + 2]

    os.makedirs(docs_dir, exist_ok=True)
    collection = get_collection(reindex=reindex)

    print(f"Carpeta de PDFs: {docs_dir}")
    print("Indexando PDFs (chunking por oraciones, incremental)...")
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
