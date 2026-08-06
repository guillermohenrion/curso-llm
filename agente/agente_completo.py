"""
Agente COMPLETO con LangChain + Ollama.

Junta en un solo ejemplo las piezas que vimos por separado en el curso:

    - TOOLS      : el agente decide que herramienta usar en cada paso
                     * buscar_conocimiento -> RAG sobre el corpus del curso
                     * calculadora         -> aritmetica (eval seguro)
                     * fecha_hoy           -> fecha actual
                     * guardar_nota / listar_notas -> memoria de largo plazo (bloc de notas)
    - RAZONAMIENTO: patron ReAct (Reasoning + Acting) -> piensa, actua, observa, repite
    - MEMORIA     : recuerda el historial de la conversacion (ventana de ultimos turnos)

A diferencia del RAG con tools (3_rag_tools.py), este agente es CONVERSACIONAL:
mantiene el hilo entre preguntas y puede acumular estado (las notas).

Usamos ReAct (basado en prompt) para que ande con gemma3, que no hace
"tool calling" nativo. Si usas un modelo con tools (p. ej. llama3.1), podes
migrar a create_tool_calling_agent.

Observabilidad (opcional): si hay credenciales de LangSmith en un .env, cada
corrida se registra en https://smith.langchain.com y podes ver el arbol ReAct
completo (cada Thought -> Action -> Observation, con latencias y tokens). Sin
credenciales, el agente funciona igual pero sin trazas.

Requisitos:
    - Ollama corriendo con:
          ollama pull gemma3
          ollama pull nomic-embed-text
    - pip install -r requirements-agente.txt
    - (opcional) copy .env.example .env  y completar LANGSMITH_API_KEY

Uso:
    python agente_completo.py                 # modo interactivo (recomendado)
    python agente_completo.py "¿Que es RAG? Y cuanto es 12*8?"
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Cargamos variables de entorno desde .env (busca en esta carpeta y en la raiz).
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Activamos el tracing de LangSmith si hay API key (LangChain lo envia solo).
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "curso-llm-agente")


def check_langsmith() -> bool:
    """Verifica credenciales de LangSmith y avisa si el tracing esta o no activo."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[aviso] No hay LANGSMITH_API_KEY. El agente funciona igual, "
              "pero NO se registran trazas en LangSmith.")
        print("        Configuralo en un archivo .env (ver .env.example) para "
              "habilitar la observabilidad.\n")
        return False
    try:
        from langsmith import Client
        Client()  # valida credenciales/endpoint
        print(f"[ok] Tracing de LangSmith habilitado. Proyecto: "
              f"{os.getenv('LANGSMITH_PROJECT')}\n")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] No se pudo inicializar el cliente de LangSmith: {e}\n")
        return False

CHAT_MODEL = "gemma3"
EMBED_MODEL = "nomic-embed-text"
MAX_TURNOS = 5  # cuantos turnos de la charla recordamos (ventana de memoria)

# --- Corpus de conocimiento del curso (para la tool de RAG) ---
DOCUMENTOS = [
    "RAG combina la recuperacion de documentos relevantes con la generacion de "
    "texto de un LLM, para dar respuestas fundamentadas en una base de conocimiento.",
    "Una base de datos vectorial almacena embeddings y busca los vectores mas "
    "similares a una consulta usando la distancia coseno.",
    "Ollama permite correr modelos de lenguaje de forma local. Soporta Gemma, "
    "Llama y embeddings como nomic-embed-text.",
    "Un agente es un LLM que, ademas de responder, puede decidir usar herramientas "
    "(tools) y encadenar pasos de razonamiento para cumplir una tarea.",
    "El patron ReAct (Reasoning + Acting) alterna pensamiento y accion: el modelo "
    "piensa, elige una herramienta, observa el resultado y repite hasta responder.",
    "LangChain es un framework para construir aplicaciones con LLMs: prompts, "
    "cadenas, retrievers, memoria, agentes y herramientas.",
]

# --- Bloc de notas en memoria (estado del agente entre turnos) ---
_NOTAS: list[str] = []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def build_retriever_tool():
    """Indexa el corpus en Chroma (en memoria) y lo expone como tool de RAG."""
    docs = [Document(page_content=t, metadata={"id": f"doc{i}"})
            for i, t in enumerate(DOCUMENTOS)]
    vs = Chroma.from_documents(
        docs, embedding=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="agente_rag",
    )
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    return create_retriever_tool(
        retriever,
        name="buscar_conocimiento",
        description=(
            "Busca en la base de conocimiento del curso sobre RAG, embeddings, "
            "bases vectoriales, Ollama, agentes, ReAct y LangChain. Usala para "
            "preguntas conceptuales sobre esos temas."
        ),
    )


@tool
def calculadora(expresion: str) -> str:
    """Evalua una expresion aritmetica simple, por ejemplo '12 * (3 + 4)'."""
    permitido = set("0123456789+-*/(). ")
    expr = "".join(ch for ch in expresion if ch in permitido).strip()
    if not expr:
        return "No encontre una expresion aritmetica valida."
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return f"Error al evaluar '{expr}': {e}"


@tool
def fecha_hoy(_: str = "") -> str:
    """Devuelve la fecha de hoy en formato YYYY-MM-DD."""
    return _dt.date.today().isoformat()


@tool
def guardar_nota(texto: str) -> str:
    """Guarda una nota para recordarla mas tarde. Recibe el texto de la nota."""
    _NOTAS.append(texto.strip())
    return f"Nota guardada (total: {len(_NOTAS)})."


@tool
def listar_notas(_: str = "") -> str:
    """Lista todas las notas guardadas hasta ahora."""
    if not _NOTAS:
        return "No hay notas guardadas."
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(_NOTAS))


# ---------------------------------------------------------------------------
# Prompt ReAct (con historial de conversacion)
# ---------------------------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """Sos un asistente util del curso de LLMs. Respondé en español.
Tenés acceso a estas herramientas:

{tools}

Usá EXACTAMENTE este formato:

Question: la pregunta de entrada
Thought: pensá que hacer
Action: la herramienta a usar, una de [{tool_names}]
Action Input: la entrada para la herramienta
Observation: el resultado de la herramienta
... (Thought/Action/Action Input/Observation se pueden repetir)
Thought: ya se la respuesta final
Final Answer: la respuesta final para el usuario

Reglas:
- Si la pregunta es conceptual sobre los temas del curso, usá 'buscar_conocimiento'.
- Si hay una cuenta, usá 'calculadora'. No calcules de memoria.
- Si el usuario pide recordar algo, usá 'guardar_nota'; para repasarlas, 'listar_notas'.
- Si ya sabés la respuesta sin herramientas, pasá directo a 'Final Answer'.

Historial de la conversacion (puede estar vacio):
{chat_history}

Empezá!

Question: {input}
Thought:{agent_scratchpad}"""
)


def build_agent() -> AgentExecutor:
    """Arma el agente ReAct: LLM + tools + prompt, envuelto en un executor."""
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)
    tools = [
        build_retriever_tool(),
        calculadora,
        fecha_hoy,
        guardar_nota,
        listar_notas,
    ]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                # muestra el razonamiento paso a paso
        handle_parsing_errors=True,  # si el LLM no respeta el formato, reintenta
        max_iterations=6,            # limite de vueltas, evita loops infinitos
    )


def formatear_historial(historial: list[tuple[str, str]]) -> str:
    """Convierte los ultimos turnos (humano, asistente) en texto para el prompt."""
    if not historial:
        return "(sin historial todavia)"
    lineas = []
    for humano, ia in historial[-MAX_TURNOS:]:
        lineas.append(f"Usuario: {humano}")
        lineas.append(f"Asistente: {ia}")
    return "\n".join(lineas)


def main() -> None:
    tracing_on = check_langsmith()
    executor = build_agent()
    historial: list[tuple[str, str]] = []

    def atender(pregunta: str) -> str:
        # run_name/tags/metadata aparecen en LangSmith para filtrar y organizar trazas.
        config = {
            "run_name": "agente_completo",
            "tags": ["curso-llm", "agente", "react"],
            "metadata": {"turno": len(historial) + 1},
        }
        salida = executor.invoke(
            {"input": pregunta, "chat_history": formatear_historial(historial)},
            config=config,
        )
        respuesta = salida["output"]
        historial.append((pregunta, respuesta))  # se suma a la memoria
        if tracing_on:
            print(f"\n[LangSmith] Traza registrada en https://smith.langchain.com "
                  f"(proyecto: {os.getenv('LANGSMITH_PROJECT')})")
        return respuesta

    argv = sys.argv[1:]
    if argv:
        pregunta = " ".join(argv)
        print(f"\n=== Pregunta: {pregunta} ===\n")
        print("\n--- Respuesta final ---")
        print(atender(pregunta))
        return

    print("Agente completo (RAG + tools + memoria). Escribí 'salir' para terminar.")
    print("Probá: '¿Que es un agente?', 'Cuanto es 15*7?', "
          "'Recorda que la clase es el martes', 'Que notas tengo?'\n")
    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            print(atender(pregunta))
            print()


if __name__ == "__main__":
    main()
