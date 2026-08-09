"""
Orquestacion de agentes con Hermes (patron supervisor / router), version MCP.

Hermes no se importa como libreria: es un agente autonomo (CLI) con su propio
orquestador. Para sumarle agentes especializados propios, la via soportada es
exponerlos como TOOLS de un servidor MCP y dejar que Hermes decida cual invocar
segun la pregunta, leyendo la descripcion (docstring) de cada tool.

Este script expone tres agentes especializados -los mismos que
orquestacion/multiagente_langgraph.py- pero como tools MCP en vez de nodos de un
grafo de LangGraph:
    - resolver_calculo    -> agente 'matematico'
    - traducir_texto      -> agente 'traductor'
    - explicar_concepto   -> agente 'explicador'

Quien hace de "supervisor" aca es Hermes: ve el catalogo de tools (via 'list_tools'
del protocolo MCP) y, para cada pedido, elige cual usar. Este script no rutea nada,
solo declara las capacidades disponibles.

Requisitos:
    - Ollama con el modelo de chat:  ollama pull gemma3
    - pip install -r requirements-hermes.txt

Probar sin Hermes (inspector oficial de MCP):
    mcp dev mcp_server_agentes.py

Conectar a Hermes:
    ver README.md de esta carpeta (registrar este script como servidor MCP,
    transport stdio).
"""
from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

CHAT_MODEL = "gemma3"

# Un unico LLM local, reutilizado por los tres agentes.
llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)

mcp = FastMCP("agentes-locales")


def _eval_seguro(expresion: str) -> str | None:
    """Evalua una expresion aritmetica simple, o None si no es evaluable.

    Misma mini-herramienta que orquestacion/multiagente_langgraph.py: evita
    depender del LLM para aritmetica basica, que suele fallar en modelos chicos.
    """
    m = re.search(r"[0-9+\-*/(). ]{3,}", expresion)
    if not m:
        return None
    expr = m.group(0).strip()
    if not set(expr) <= set("0123456789+-*/(). "):
        return None
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 (sandbox minimo)
    except Exception:  # noqa: BLE001
        return None


@mcp.tool()
def resolver_calculo(pregunta: str) -> str:
    """Resuelve calculos y problemas aritmeticos (sumas, productos, expresiones
    con parentesis, etc). Usala cuando la pregunta sea un calculo matematico."""
    resultado = _eval_seguro(pregunta)
    if resultado is not None:
        return f"El resultado del calculo es: {resultado}"
    system = SystemMessage(content="Sos un agente que resuelve problemas matematicos. "
                                   "Explicá el paso a paso y dá el resultado final.")
    msg = llm.invoke([system, HumanMessage(content=pregunta)])
    return msg.content


@mcp.tool()
def traducir_texto(pregunta: str) -> str:
    """Traduce texto entre idiomas. Usala cuando pidan traducir algo a otro idioma."""
    system = SystemMessage(content="Sos un agente traductor. Traducí lo que pida el usuario. "
                                   "Devolvé SOLO la traduccion, sin explicaciones.")
    msg = llm.invoke([system, HumanMessage(content=pregunta)])
    return msg.content


@mcp.tool()
def explicar_concepto(pregunta: str) -> str:
    """Explica conceptos generales o tecnicos de forma clara y breve. Usala para
    cualquier pregunta conceptual que no sea un calculo ni una traduccion."""
    system = SystemMessage(content="Sos un agente que explica conceptos de forma clara y breve, "
                                   "en español. Respondé de manera didactica.")
    msg = llm.invoke([system, HumanMessage(content=pregunta)])
    return msg.content


if __name__ == "__main__":
    mcp.run(transport="stdio")
