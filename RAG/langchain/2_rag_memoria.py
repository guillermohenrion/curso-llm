"""
RAG CONVERSACIONAL con MEMORIA AVANZADA (LangChain).

Suma dos cosas sobre el RAG basico:

1. **History-aware retriever**: antes de buscar, reformula la pregunta del
   usuario teniendo en cuenta el historial de la charla. Asi funcionan las
   preguntas de seguimiento tipo "¿y eso para que sirve?" (que sin contexto
   no se podrian buscar bien).

2. **Memoria por sesion con recorte (trimming)**: guardamos el historial de
   cada sesion en memoria y lo recortamos a los ultimos N mensajes con
   `trim_messages`, para no crecer indefinidamente (memoria "avanzada":
   acotada por ventana en vez de todo el historial).

Uso:
    python 2_rag_memoria.py            # modo interactivo (conversacion)
    python 2_rag_memoria.py "¿Que es RAG?"   # una pregunta y listo
"""
from __future__ import annotations

import sys

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

from comun import get_llm, get_retriever, leer_pregunta_o_argv

MAX_MENSAJES = 6  # ventana de memoria: ultimos 6 mensajes (3 turnos user+assistant)

# Prompt para reformular la pregunta usando el historial (contextualizacion).
CONTEXTUALIZAR = ChatPromptTemplate.from_messages([
    ("system",
     "Dada la conversacion y la ultima pregunta del usuario, reformulala como "
     "una pregunta autonoma que se entienda sin el historial. NO la respondas, "
     "solo reformulala (o devolvela igual si ya es autonoma)."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt para responder usando el contexto recuperado.
RESPONDER = ChatPromptTemplate.from_messages([
    ("system",
     "Sos un asistente que responde SOLO con la informacion del contexto.\n"
     "Si el contexto no alcanza, decilo.\n\nContexto:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Almacen de historiales por sesion (en memoria).
_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Devuelve (creando si hace falta) el historial recortado de una sesion."""
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    history = _STORE[session_id]
    # Memoria avanzada: recortamos a los ultimos MAX_MENSAJES mensajes.
    recortados = trim_messages(
        history.messages,
        max_tokens=MAX_MENSAJES,
        token_counter=len,          # contamos por cantidad de mensajes
        strategy="last",            # nos quedamos con los mas recientes
        include_system=False,
        start_on="human",
    )
    history.clear()
    history.add_messages(recortados)
    return history


def build_conversational_rag():
    """Arma la cadena de RAG conversacional con memoria por sesion."""
    llm = get_llm()
    retriever = get_retriever(k=2)

    # 1. Retriever que reformula la pregunta con el historial.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZAR
    )

    # 2. Cadena que "rellena" los documentos en el prompt y genera la respuesta.
    qa_chain = create_stuff_documents_chain(llm, RESPONDER)

    # 3. Cadena RAG completa: recuperar (con historial) + responder.
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 4. Envolvemos con memoria por sesion.
    conversational = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational


def preguntar(chain, pregunta: str, session_id: str = "demo") -> str:
    """Hace una pregunta dentro de una sesion; session_id identifica que historial usar."""
    salida = chain.invoke(
        {"input": pregunta},
        config={"configurable": {"session_id": session_id}},  # asocia esta llamada a la sesion
    )
    return salida["answer"]


def main() -> None:
    chain = build_conversational_rag()

    if len(sys.argv) > 1:
        pregunta = leer_pregunta_o_argv("¿Que es RAG?")
        print(f"\n=== {pregunta} ===")
        print(preguntar(chain, pregunta))
        return

    # Modo interactivo: la memoria mantiene el hilo de la conversacion.
    print("RAG conversacional. Escribi 'salir' para terminar.")
    print("Probá una pregunta y despues un seguimiento tipo '¿y para que sirve?'\n")
    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            print(preguntar(chain, pregunta))
            print()


if __name__ == "__main__":
    main()
