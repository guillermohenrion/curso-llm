"""
RAG BASICO con LangChain (LCEL).

Version mas simple posible: una sola pregunta -> se recuperan documentos
relevantes -> se arma un prompt con ese contexto -> el LLM responde.

No hay memoria, ni herramientas, ni agente. Es el "hola mundo" de RAG.

Flujo (LCEL):
    {contexto: retriever, pregunta: passthrough} -> prompt -> LLM -> texto

Uso:
    python 1_rag_basico.py
    python 1_rag_basico.py "¿Que es una base de datos vectorial?"
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from comun import get_llm, get_retriever, format_docs, leer_pregunta_o_argv

PROMPT = ChatPromptTemplate.from_template(
    "Sos un asistente que responde SOLO con la informacion del contexto.\n"
    "Si el contexto no alcanza para responder, decilo explicitamente.\n\n"
    "Contexto:\n{context}\n\n"
    "Pregunta: {question}\n"
    "Respuesta:"
)


def build_chain():
    """Construye la cadena de RAG basico con LCEL."""
    retriever = get_retriever(k=2)
    llm = get_llm()

    # El operador | encadena componentes (Runnables).
    # Para 'context' recuperamos docs y los formateamos; 'question' pasa tal cual.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def main() -> None:
    pregunta = leer_pregunta_o_argv("¿Que es RAG y para que sirve?")
    chain = build_chain()

    print(f"\n=== Pregunta: {pregunta} ===\n")
    respuesta = chain.invoke(pregunta)
    print(respuesta)


if __name__ == "__main__":
    main()
