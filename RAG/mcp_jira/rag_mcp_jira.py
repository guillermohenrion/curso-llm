"""
RAG + MCP de JIRA: un agente que combina una base de conocimiento (RAG) con las
herramientas de Jira expuestas por un servidor MCP (Model Context Protocol).

Idea: partimos del RAG simple (recuperar de una base vectorial) y le sumamos, via
MCP, acceso EN VIVO a Jira. El agente decide en cada pregunta si:
    - buscar en la base de conocimiento del curso  (tool RAG local), o
    - consultar Jira (buscar issues, ver un ticket, etc.) via las tools del MCP.

Componentes:
    - LLM con soporte de tools (Ollama, p. ej. llama3.1)  ->  decide que tool usar
    - Retriever tool (Chroma + OllamaEmbeddings)          ->  RAG del curso
    - MCP 'mcp-atlassian'                                 ->  tools de Jira
      (se lanza como subproceso con `uvx mcp-atlassian`, transport stdio)

Requisitos:
    - Ollama corriendo con un modelo que soporte tools:
          ollama pull llama3.1
          ollama pull nomic-embed-text
    - `uv` instalado (para `uvx`):  https://docs.astral.sh/uv/
    - Un token de Jira (Cloud):  https://id.atlassian.com/manage-profile/security/api-tokens
    - pip install -r requirements-mcp-jira.txt
    - copy .env.example .env   y completar JIRA_URL / JIRA_USERNAME / JIRA_API_TOKEN

Uso:
    python rag_mcp_jira.py "Listá mis issues abiertos en el proyecto ABC"
    python rag_mcp_jira.py "¿Que es una base de datos vectorial?"   # usa el RAG
    python rag_mcp_jira.py                                          # interactivo

Nota: si no configurás Jira, el agente igual arranca pero solo con la tool de RAG.

Gotchas reales encontrados al correr esto (por eso algunas lineas de mas abajo
tienen pinta de "parche"):
    - 'pip install' puede traer mcp>=2.0, incompatible con langchain-mcp-adapters
      0.1.x -> fijar mcp<2.0 en requirements-mcp-jira.txt.
    - uvx puede fallar con "invalid peer certificate" si un antivirus/firewall
      intercepta TLS -> se lanza con el flag --native-tls (ver load_jira_tools).
    - El MCP de Atlassian expone ~35 tools; sus esquemas no entran en el
      contexto default de Ollama (2048 tokens) y el modelo deja de ver las
      tools -> se sube num_ctx explicitamente (ver build_agent).
    - Jira Cloud rechaza JQL sin restricciones ("unbounded query") -> el
      prompt le pide al modelo usar una JQL acotada (por proyecto o fecha).
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(__file__)
load_dotenv()
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Modelo con soporte de tools (gemma3 no hace tool-calling; usar llama3.1 u otro).
TOOL_MODEL = os.getenv("OLLAMA_TOOL_MODEL", "llama3.1")
EMBED_MODEL = "nomic-embed-text"

# Verbos que indican ESCRITURA (modificacion/insercion): si aparecen en el nombre
# de una tool, la descartamos. Filtro conservador: ante la duda, se excluye.
WRITE_HINTS = (
    "create", "update", "delete", "add", "remove", "edit", "transition",
    "assign", "move", "upload", "link", "unlink", "set", "archive", "restore",
    "comment", "write", "insert", "modify", "put", "post", "patch", "batch_create",
)
# Verbos que indican LECTURA: la tool tiene que tener al menos uno de estos.
READ_HINTS = (
    "get", "search", "list", "read", "download", "fetch", "describe",
    "lookup", "view", "summary",
)


def es_solo_lectura(nombre: str) -> bool:
    """True solo si el nombre de la tool sugiere lectura y NO escritura."""
    n = nombre.lower()
    if any(w in n for w in WRITE_HINTS):
        return False
    return any(r in n for r in READ_HINTS)

# --- Corpus de ejemplo para la tool de RAG (base de conocimiento del curso) ---
DOCUMENTOS = [
    "RAG combina la recuperacion de documentos relevantes con la generacion de "
    "texto de un LLM, para dar respuestas fundamentadas en una base de conocimiento.",
    "Una base de datos vectorial almacena embeddings y busca los vectores mas "
    "similares a una consulta usando la distancia coseno.",
    "Ollama permite correr modelos de lenguaje de forma local. Soporta Gemma, "
    "Llama y embeddings como nomic-embed-text.",
    "MCP (Model Context Protocol) es un protocolo estandar para que los LLMs se "
    "conecten a herramientas y fuentes de datos externas mediante servidores MCP.",
    "El MCP de Atlassian (mcp-atlassian) expone herramientas para interactuar con "
    "Jira y Confluence: buscar issues con JQL, leer un ticket, crear o comentar, etc.",
]


def build_retriever_tool():
    """Crea la base vectorial en memoria y la envuelve como una tool de RAG."""
    docs = [Document(page_content=t, metadata={"id": f"doc{i}"})
            for i, t in enumerate(DOCUMENTOS)]
    vs = Chroma.from_documents(
        docs, embedding=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="mcp_jira_rag",
    )
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    return create_retriever_tool(
        retriever,
        name="buscar_conocimiento",
        description=(
            "Busca en la base de conocimiento del curso sobre RAG, embeddings, "
            "bases vectoriales, Ollama y MCP. Usala para preguntas conceptuales."
        ),
    )


async def load_jira_tools() -> list:
    """Conecta al MCP de Atlassian y devuelve sus tools (o [] si no hay credenciales)."""
    jira_url = os.getenv("JIRA_URL")
    jira_user = os.getenv("JIRA_USERNAME")
    jira_token = os.getenv("JIRA_API_TOKEN")
    if not (jira_url and jira_user and jira_token):
        print("[aviso] Faltan credenciales de Jira en .env: se usa SOLO la tool de RAG.\n")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("[aviso] Falta 'langchain-mcp-adapters'. Instalá requirements-mcp-jira.txt.\n")
        return []

    # Se lanza el servidor MCP 'mcp-atlassian' como subproceso (via uvx) por stdio.
    # PROTECCION 1: READ_ONLY_MODE=true hace que el propio servidor MCP deshabilite
    # todas las operaciones de escritura (crear/editar/borrar/comentar/transicionar).
    client = MultiServerMCPClient({
        "jira": {
            "command": "uvx",
            # --native-tls: usa el almacen de certificados del sistema en vez del
            # propio de uv, necesario si un antivirus/firewall intercepta TLS.
            "args": ["--native-tls", "mcp-atlassian"],
            "transport": "stdio",
            "env": {
                "JIRA_URL": jira_url,
                "JIRA_USERNAME": jira_user,
                "JIRA_API_TOKEN": jira_token,
                "READ_ONLY_MODE": "true",
            },
        }
    })
    try:
        tools = await client.get_tools()
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] No se pudo conectar al MCP de Jira ({e}). Se usa solo RAG.\n")
        return []

    # PROTECCION 2 (defensa en profundidad): ademas filtramos del lado del cliente
    # y dejamos SOLO tools de lectura, descartando cualquiera con verbos de escritura.
    de_lectura = [t for t in tools if es_solo_lectura(t.name)]
    # Si una tool falla (p. ej. JQL rechazada por Jira), que el error vuelva como
    # Observation en vez de tirar abajo el AgentExecutor: asi el agente puede leer
    # el motivo y reintentar con una consulta mejor.
    for t in de_lectura:
        t.handle_tool_error = True
    descartadas = [t.name for t in tools if not es_solo_lectura(t.name)]
    if descartadas:
        print(f"[seguridad] Tools de escritura descartadas ({len(descartadas)}): "
              f"{', '.join(sorted(descartadas))}")
    print(f"[ok] MCP de Jira conectado: {len(de_lectura)} tools de LECTURA habilitadas.\n")
    return de_lectura


def build_agent(tools):
    # num_ctx: el default de Ollama (2048) no alcanza para los esquemas de las
    # ~35 tools del MCP de Atlassian + RAG; con contexto chico el modelo ni
    # siquiera ve las tools y termina alucinando una respuesta en texto plano.
    llm = ChatOllama(model=TOOL_MODEL, temperature=0.0, num_ctx=16384)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sos un asistente que ayuda con el curso de LLMs y con Jira.\n"
         "- Para preguntas conceptuales (RAG, embeddings, MCP, etc.) usá la tool "
         "'buscar_conocimiento'.\n"
         "- Para consultas sobre issues, tickets, proyectos o sprints, usá las tools "
         "de Jira (MCP). El acceso a Jira es de SOLO LECTURA: NO podés crear, editar, "
         "borrar, comentar ni transicionar nada. Si te piden modificar algo, aclaralo.\n"
         "- Nunca inventes una clave de issue (p. ej. 'PROJ-123'). Si no la sabés, "
         "primero usá 'jira_search'. Jira exige una JQL ACOTADA (no acepta una consulta "
         "sin restricciones): usá algo como \"created >= -365d ORDER BY created DESC\" "
         "para traer cualquier issue reciente, y recién ahí consultá el detalle si hace falta.\n"
         "Respondé en español y citá de donde sacaste la informacion."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True,
                         handle_parsing_errors=True, max_iterations=6)


async def responder(executor, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===\n")
    salida = await executor.ainvoke({"input": pregunta})
    print("\n--- Respuesta final ---")
    print(salida["output"])


async def main_async() -> None:
    tools = [build_retriever_tool()]
    tools += await load_jira_tools()
    executor = build_agent(tools)

    argv = sys.argv[1:]
    if argv:
        await responder(executor, " ".join(argv))
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
                await responder(executor, pregunta)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
