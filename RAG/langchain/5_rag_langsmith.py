"""
RAG con TRACING en LANGSMITH.

Toma el RAG basico y le agrega **observabilidad**: cada ejecucion se registra
en LangSmith (https://smith.langchain.com), donde podes ver el arbol de pasos
(retriever, prompt, LLM), latencias, tokens y las entradas/salidas de cada
componente. Muy util para depurar y evaluar aplicaciones con LLMs.

Como funciona el tracing:
    LangChain envia trazas automaticamente si estan seteadas estas variables de
    entorno (las leemos de un archivo .env):
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY=ls_...
        LANGSMITH_PROJECT=curso-llm-rag   (opcional; agrupa las trazas)
        LANGSMITH_ENDPOINT=https://api.smith.langchain.com  (opcional)

Setup:
    1. Crear cuenta gratis en https://smith.langchain.com
    2. Generar una API key en Settings.
    3. Copiar .env.example a .env y completar LANGSMITH_API_KEY.
       (o exportar las variables de entorno a mano)
    4. pip install -r requirements-langchain.txt

Uso:
    python 5_rag_langsmith.py
    python 5_rag_langsmith.py "¿Que es una base de datos vectorial?"

Al terminar, el script imprime el link para ver la traza en LangSmith.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from comun import get_llm, get_retriever, format_docs, leer_pregunta_o_argv

# Cargamos variables de entorno desde .env (busca en esta carpeta y en la raiz).
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Aseguramos que el tracing este activado si hay API key.
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "curso-llm-rag")

PROMPT = ChatPromptTemplate.from_template(
    "Respondé usando SOLO el contexto. Si no alcanza, decilo.\n\n"
    "Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
)


def check_langsmith() -> bool:
    """Verifica que haya credenciales de LangSmith y (si se puede) la conexion."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[aviso] No hay LANGSMITH_API_KEY. El RAG va a funcionar igual, "
              "pero NO se van a registrar trazas en LangSmith.")
        print("        Configuralo en un archivo .env (ver .env.example) para "
              "habilitar el tracing.\n")
        return False
    try:
        from langsmith import Client
        Client()  # valida credenciales/endpoint
        proyecto = os.getenv("LANGSMITH_PROJECT")
        print(f"[ok] Tracing de LangSmith habilitado. Proyecto: {proyecto}\n")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] No se pudo inicializar el cliente de LangSmith: {e}\n")
        return False


def build_chain():
    """Misma cadena LCEL que el RAG basico (Seccion 1); el tracing se agrega solo via env vars."""
    retriever = get_retriever(k=2)
    llm = get_llm()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def main() -> None:
    tracing_on = check_langsmith()
    pregunta = leer_pregunta_o_argv("¿Que es RAG y para que sirve?")
    chain = build_chain()

    print(f"=== Pregunta: {pregunta} ===\n")

    # Pasamos metadata/tags: aparecen en LangSmith para filtrar y organizar trazas.
    config = {
        "run_name": "rag_basico_langsmith",
        "tags": ["curso-llm", "rag", "clase"],
        "metadata": {"version": "5-langsmith"},
    }
    respuesta = chain.invoke(pregunta, config=config)
    print(respuesta)

    if tracing_on:
        proyecto = os.getenv("LANGSMITH_PROJECT")
        print(f"\n[LangSmith] Trazá registrada. Miralo en: "
              f"https://smith.langchain.com  (proyecto: {proyecto})")


if __name__ == "__main__":
    main()
