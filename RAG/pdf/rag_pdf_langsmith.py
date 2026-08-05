"""
RAG sobre una carpeta de PDFs, con LANGCHAIN + observabilidad en LANGSMITH.

Misma idea que rag_pdf.py (indexa PDFs en ChromaDB y consulta), pero construido
con LangChain (LCEL). Al estar seteadas las variables de entorno de LangSmith,
CADA ejecucion se traza automaticamente en https://smith.langchain.com: se ve el
arbol retriever -> prompt -> LLM, latencias, tokens y entradas/salidas.

Comparte la carpeta docs/ con rag_pdf.py (poné ahi tus PDFs), pero usa su propio
indice persistente (chroma_pdf_lc/) para no mezclarse con el otro ejemplo.

Setup de LangSmith:
    1. Cuenta gratis en https://smith.langchain.com y una API key (Settings).
    2. copy .env.example .env  y completar LANGSMITH_API_KEY.

Requisitos:
    - Ollama corriendo:  ollama pull gemma3 ; ollama pull nomic-embed-text
    - pip install -r requirements-pdf-langsmith.txt

Uso:
    python rag_pdf_langsmith.py "¿De que trata el documento?"
    python rag_pdf_langsmith.py --reindex "..."   # reconstruye el indice
    python rag_pdf_langsmith.py                    # interactivo
"""
from __future__ import annotations

import os
import sys
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_pdf_lc")
COLLECTION_NAME = "pdf_docs_lc"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_K = 4

# --- LangSmith: cargar credenciales de .env y activar tracing ---
load_dotenv()
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "curso-llm-rag-pdf")

PROMPT = ChatPromptTemplate.from_template(
    "Respondé la pregunta usando SOLO el contexto extraido de los PDFs. "
    "Si el contexto no alcanza, decilo. Citá las fuentes.\n\n"
    "Contexto:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)


def check_langsmith() -> bool:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[aviso] No hay LANGSMITH_API_KEY: el RAG funciona igual, pero NO se "
              "registran trazas en LangSmith. Configuralo en .env para habilitarlo.\n")
        return False
    try:
        from langsmith import Client
        Client()
        print(f"[ok] Tracing de LangSmith habilitado. Proyecto: "
              f"{os.getenv('LANGSMITH_PROJECT')}\n")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] No se pudo inicializar LangSmith: {e}\n")
        return False


def format_docs(docs) -> str:
    bloques = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "?"))
        page = d.metadata.get("page")
        bloques.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n".join(bloques)


def build_vectorstore(reindex: bool = False) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if reindex and os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Indexamos solo si el indice esta vacio (o si se pidio --reindex).
    if vs._collection.count() == 0:
        os.makedirs(DOCS_DIR, exist_ok=True)
        pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
        if not pdfs:
            print(f"[aviso] No hay PDFs en {DOCS_DIR}. Poné tus archivos ahi.")
            return vs
        print(f"Indexando {len(pdfs)} PDF(s) de {DOCS_DIR}...")
        docs = PyPDFDirectoryLoader(DOCS_DIR).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vs.add_documents(chunks)
        print(f"Indexados {len(chunks)} chunks.\n")
    else:
        print(f"Indice existente con {vs._collection.count()} chunks "
              f"(usá --reindex para reconstruir).\n")
    return vs


def build_chain(vs: Chroma):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def consultar(chain, pregunta: str, tracing_on: bool) -> None:
    print(f"\n=== Pregunta: {pregunta} ===\n")
    config = {
        "run_name": "rag_pdf_langsmith",
        "tags": ["curso-llm", "rag", "pdf"],
        "metadata": {"variante": "pdf-langchain-langsmith"},
    }
    print(chain.invoke(pregunta, config=config))
    if tracing_on:
        print(f"\n[LangSmith] Traza registrada en https://smith.langchain.com "
              f"(proyecto: {os.getenv('LANGSMITH_PROJECT')})")


def main() -> None:
    argv = sys.argv[1:]
    reindex = "--reindex" in argv
    argv = [a for a in argv if a != "--reindex"]

    tracing_on = check_langsmith()
    vs = build_vectorstore(reindex=reindex)
    if vs._collection.count() == 0:
        print("No hay nada indexado. Agregá PDFs en docs/ y volvé a correr.")
        return

    chain = build_chain(vs)

    if argv:
        consultar(chain, " ".join(argv), tracing_on)
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
                consultar(chain, pregunta, tracing_on)


if __name__ == "__main__":
    main()
