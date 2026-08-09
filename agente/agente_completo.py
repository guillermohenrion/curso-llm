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

# Cargamos variables de entorno desde .env. Hacemos dos llamadas a proposito:
#   1) load_dotenv()             -> busca un .env en el directorio actual (cwd).
#   2) load_dotenv(dotenv_path=) -> busca el .env que esta AL LADO de este archivo,
#                                   asi funciona aunque lo corras desde otra carpeta.
# load_dotenv NO pisa variables ya definidas en el entorno del sistema.
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Tracing de LangSmith: LangChain envia las trazas SOLO si estas variables de
# entorno estan seteadas. No hay que instrumentar el codigo a mano; con tener
# LANGSMITH_API_KEY alcanza. Aca la damos por activada (LANGSMITH_TRACING=true)
# si hay API key, y fijamos un nombre de proyecto por defecto (donde se agrupan
# las corridas en la UI de LangSmith). setdefault no pisa el valor si ya venia del .env.
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

CHAT_MODEL = "gemma3"            # LLM que razona y redacta (el "cerebro" del agente)
EMBED_MODEL = "nomic-embed-text"  # modelo que convierte texto en vectores (solo para el RAG)
MAX_TURNOS = 5  # cuantos turnos de la charla recordamos (ventana de memoria; ver formatear_historial)

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
    # 1) Texto plano -> Document de LangChain (formato que espera Chroma).
    docs = [Document(page_content=t, metadata={"id": f"doc{i}"})
            for i, t in enumerate(DOCUMENTOS)]
    # 2) from_documents embebe cada texto (con OllamaEmbeddings) y arma el indice
    #    vectorial. Es EN MEMORIA: se reconstruye en cada corrida (suficiente para clase).
    vs = Chroma.from_documents(
        docs, embedding=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="agente_rag",
    )
    # 3) as_retriever con k=2 -> ante una consulta, devuelve los 2 chunks mas similares.
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    # 4) create_retriever_tool envuelve el retriever como una TOOL que el agente puede
    #    elegir. OJO: el 'name' y sobre todo la 'description' son lo que el LLM lee para
    #    decidir cuando usar esta tool -> conviene que sean claros y especificos.
    return create_retriever_tool(
        retriever,
        name="buscar_conocimiento",
        description=(
            "Busca en la base de conocimiento del curso sobre RAG, embeddings, "
            "bases vectoriales, Ollama, agentes, ReAct y LangChain. Usala para "
            "preguntas conceptuales sobre esos temas."
        ),
    )


# El decorador @tool convierte una funcion en una tool de LangChain: usa el NOMBRE
# de la funcion como nombre de la tool y su DOCSTRING como descripcion (lo que el
# agente lee para decidir si la usa). En ReAct, la tool recibe UN solo string
# (el "Action Input"), por eso todas toman un unico parametro de texto.
@tool
def calculadora(expresion: str) -> str:
    """Evalua una expresion aritmetica simple, por ejemplo '12 * (3 + 4)'."""
    # Sandbox minimo: nos quedamos SOLO con caracteres de aritmetica, para no
    # permitir que un 'eval' ejecute codigo arbitrario que venga en el texto.
    permitido = set("0123456789+-*/(). ")
    expr = "".join(ch for ch in expresion if ch in permitido).strip()
    if not expr:
        return "No encontre una expresion aritmetica valida."
    try:
        # __builtins__ vacio -> se deshabilitan funciones como open/import dentro del eval.
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return f"Error al evaluar '{expr}': {e}"


# El parametro '_' existe solo porque el formato ReAct siempre pasa un Action Input,
# aunque esta tool no necesite entrada. Lo ignoramos.
@tool
def fecha_hoy(_: str = "") -> str:
    """Devuelve la fecha de hoy en formato YYYY-MM-DD."""
    return _dt.date.today().isoformat()


# guardar_nota + listar_notas comparten la lista _NOTAS (estado en memoria del
# proceso): muestran como un agente puede ACUMULAR informacion entre turnos y
# consultarla despues. Al cerrar el programa, las notas se pierden.
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
# Este prompt define el "protocolo" que sigue el agente. Los {placeholders} los
# rellena LangChain automaticamente en cada vuelta del bucle:
#   {tools}            -> nombre + descripcion de cada tool disponible
#   {tool_names}       -> lista de nombres validos para la linea 'Action'
#   {chat_history}     -> historial de la conversacion (lo armamos nosotros; ver main)
#   {input}            -> la pregunta actual del usuario
#   {agent_scratchpad} -> el borrador de Thought/Action/Observation de ESTE turno,
#                         que el executor va acumulando vuelta a vuelta
# El modelo NO ejecuta tools: solo escribe texto siguiendo el formato; el executor
# parsea 'Action'/'Action Input', corre la tool y pega el resultado como 'Observation'.
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
- Si la pregunta tiene VARIAS partes o pide VARIAS tareas (p. ej. "buscá X, calculá Y y
  guardá Z"), resolvé cada parte con su propia Action, UNA POR VEZ. No pases a
  'Final Answer' hasta haber cubierto todas las partes pedidas.
- Nunca repitas la misma Action con el mismo Action Input si ya tenés esa Observation:
  pasá a la siguiente parte pendiente o a 'Final Answer'.

Historial de la conversacion (puede estar vacio):
{chat_history}

Empezá!

Question: {input}
Thought:{agent_scratchpad}"""
)


def build_agent() -> AgentExecutor:
    """Arma el agente ReAct: LLM + tools + prompt, envuelto en un executor."""
    # temperature=0.0 -> respuestas mas deterministas (util para que respete el formato ReAct).
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)
    # 5 tools: es la cantidad recomendada (pocas tools = el agente se confunde menos).
    tools = [
        build_retriever_tool(),
        calculadora,
        fecha_hoy,
        guardar_nota,
        listar_notas,
    ]
    # create_react_agent une LLM + tools + prompt en un agente que "habla" en formato ReAct.
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    # El AgentExecutor es quien corre el bucle de verdad (Thought->Action->Observation).
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                # imprime cada paso del razonamiento en la consola
        handle_parsing_errors=True,  # si el LLM no respeta el formato, le avisa y reintenta (no crashea)
        max_iterations=6,            # tope de vueltas del bucle: evita loops infinitos
    )


def formatear_historial(historial: list[tuple[str, str]]) -> str:
    """Convierte los ultimos turnos (humano, asistente) en texto para el prompt."""
    if not historial:
        return "(sin historial todavia)"
    # historial[-MAX_TURNOS:] -> ventana deslizante: solo los ultimos N turnos, para
    # que el prompt no crezca sin limite (mas historial = mas tokens = mas lento/caro).
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
        # invoke dispara el bucle ReAct completo. Le pasamos la pregunta y el historial
        # ya formateado (esos dos {placeholders} son los que faltaban en REACT_PROMPT).
        salida = executor.invoke(
            {"input": pregunta, "chat_history": formatear_historial(historial)},
            config=config,
        )
        respuesta = salida["output"]
        historial.append((pregunta, respuesta))  # se suma a la memoria para el proximo turno
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
