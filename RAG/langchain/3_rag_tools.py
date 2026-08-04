"""
RAG con TOOLS (agente ReAct de LangChain).

En vez de una cadena fija, ahora hay un AGENTE que decide, en cada paso, que
herramienta usar. Le damos varias tools:

    - buscar_docs : RAG (recupera de la base vectorial)   <- la clave del RAG
    - calculadora : evalua una expresion aritmetica
    - fecha_hoy   : devuelve la fecha actual

El agente razona con el patron ReAct (Reasoning + Acting): piensa, elige una
tool, observa el resultado, y repite hasta poder responder. Usamos ReAct
(basado en prompt) para que funcione con Gemma sin necesidad de un modelo con
"tool calling" nativo.

Uso:
    python 3_rag_tools.py
    python 3_rag_tools.py "¿Que es ChromaDB? Ademas, cuanto es 12*8?"
"""
from __future__ import annotations

import datetime as _dt

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from comun import get_llm, get_retriever, leer_pregunta_o_argv


# --- Tool 1: RAG (recuperacion sobre el corpus) ---
def build_retriever_tool():
    retriever = get_retriever(k=2)
    return create_retriever_tool(
        retriever,
        name="buscar_docs",
        description=(
            "Busca informacion en la base de conocimiento sobre RAG, embeddings, "
            "bases vectoriales, Ollama, ChromaDB y LangChain. Usala para cualquier "
            "pregunta conceptual sobre esos temas."
        ),
    )


# --- Tool 2: calculadora ---
@tool
def calculadora(expresion: str) -> str:
    """Evalua una expresion aritmetica simple, por ejemplo '12 * (3 + 4)'."""
    permitido = set("0123456789+-*/(). ")
    if not set(expresion) <= permitido:
        return "Error: la expresion tiene caracteres no permitidos."
    try:
        return str(eval(expresion, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"Error al evaluar: {e}"


# --- Tool 3: fecha actual ---
@tool
def fecha_hoy(_: str = "") -> str:
    """Devuelve la fecha de hoy en formato YYYY-MM-DD."""
    return _dt.date.today().isoformat()


# Prompt ReAct (inline, para no depender de langchainhub / internet).
REACT_PROMPT = PromptTemplate.from_template(
    """Respondé la pregunta lo mejor posible. Tenés acceso a estas herramientas:

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

Empezá!

Question: {input}
Thought:{agent_scratchpad}"""
)


def build_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [build_retriever_tool(), calculadora, fecha_hoy]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,               # muestra el razonamiento paso a paso
        handle_parsing_errors=True,
        max_iterations=6,
    )


def main() -> None:
    pregunta = leer_pregunta_o_argv(
        "¿Que es una base de datos vectorial? Y de paso, cuanto es 15 * 7?"
    )
    executor = build_agent()
    print(f"\n=== Pregunta: {pregunta} ===\n")
    salida = executor.invoke({"input": pregunta})
    print("\n--- Respuesta final ---")
    print(salida["output"])


if __name__ == "__main__":
    main()
