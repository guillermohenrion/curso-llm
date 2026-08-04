"""
RAG con SKILLS (capacidades de alto nivel + router).

Una "skill" es una capacidad con nombre, descripcion y una cadena propia.
A diferencia del agente con tools (donde el LLM decide paso a paso en un bucle
ReAct), aca hay un ROUTER que clasifica la intencion del usuario y despacha a
UNA skill especializada. Es un patron mas declarativo y controlable.

Skills incluidas:
    - rag      : responde preguntas conceptuales usando la base vectorial (RAG)
    - resumir  : resume un texto que pega el usuario
    - traducir : traduce un texto al ingles
    - calcular : resuelve una operacion aritmetica

Flujo:
    pregunta -> ROUTER (elige skill) -> skill.run(pregunta) -> respuesta

Uso:
    python 4_rag_skills.py
    python 4_rag_skills.py "Traducir: los embeddings representan significado"
    python 4_rag_skills.py "¿Que es LangChain?"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from comun import get_llm, get_retriever, format_docs, leer_pregunta_o_argv


# ---------------------------------------------------------------------------
# Definicion de una skill
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    nombre: str
    descripcion: str
    run: Callable[[str], str]


# ---------------------------------------------------------------------------
# Skill 1: RAG (retrieval + generacion)
# ---------------------------------------------------------------------------
def _build_rag_skill() -> Callable[[str], str]:
    retriever = get_retriever(k=2)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Respondé usando SOLO el contexto. Si no alcanza, decilo.\n\n"
        "Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return lambda entrada: chain.invoke(entrada)


# ---------------------------------------------------------------------------
# Skill 2: resumir
# ---------------------------------------------------------------------------
def _build_resumir_skill() -> Callable[[str], str]:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Resumí el siguiente texto en 1 o 2 oraciones claras:\n\n{texto}\n\nResumen:"
    )
    chain = prompt | llm | StrOutputParser()
    return lambda entrada: chain.invoke({"texto": entrada})


# ---------------------------------------------------------------------------
# Skill 3: traducir
# ---------------------------------------------------------------------------
def _build_traducir_skill() -> Callable[[str], str]:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Traducí el siguiente texto al ingles. Devolvé SOLO la traduccion.\n\n"
        "Texto: {texto}\nTraduccion:"
    )
    chain = prompt | llm | StrOutputParser()
    return lambda entrada: chain.invoke({"texto": entrada})


# ---------------------------------------------------------------------------
# Skill 4: calcular
# ---------------------------------------------------------------------------
def _calcular(entrada: str) -> str:
    permitido = set("0123456789+-*/(). ")
    expr = "".join(ch for ch in entrada if ch in permitido).strip()
    if not expr:
        return "No encontre una expresion aritmetica en la entrada."
    try:
        return f"Resultado: {eval(expr, {'__builtins__': {}}, {})}"
    except Exception as e:  # noqa: BLE001
        return f"Error al evaluar '{expr}': {e}"


# ---------------------------------------------------------------------------
# Registro de skills
# ---------------------------------------------------------------------------
def build_skills() -> dict[str, Skill]:
    return {
        "rag": Skill(
            "rag",
            "Responder preguntas conceptuales sobre RAG, embeddings, bases "
            "vectoriales, Ollama, ChromaDB o LangChain usando la base de conocimiento.",
            _build_rag_skill(),
        ),
        "resumir": Skill(
            "resumir",
            "Resumir un texto que provee el usuario.",
            _build_resumir_skill(),
        ),
        "traducir": Skill(
            "traducir",
            "Traducir al ingles un texto que provee el usuario.",
            _build_traducir_skill(),
        ),
        "calcular": Skill(
            "calcular",
            "Resolver una operacion o expresion aritmetica.",
            _calcular,
        ),
    }


# ---------------------------------------------------------------------------
# Router: elige la skill adecuada con el LLM
# ---------------------------------------------------------------------------
def build_router(skills: dict[str, Skill]):
    catalogo = "\n".join(f"- {s.nombre}: {s.descripcion}" for s in skills.values())
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Elegí la skill mas adecuada para atender la peticion del usuario.\n"
        "Skills disponibles:\n{catalogo}\n\n"
        "Peticion: {entrada}\n\n"
        "Respondé UNICAMENTE con el nombre de la skill (una sola palabra), sin explicar."
    )
    chain = prompt | llm | StrOutputParser()

    def router(entrada: str) -> str:
        elegido = chain.invoke({"catalogo": catalogo, "entrada": entrada}).strip().lower()
        # Nos quedamos con la primera skill cuyo nombre aparezca en la respuesta.
        for nombre in skills:
            if nombre in elegido:
                return nombre
        return "rag"  # fallback razonable

    return router


def main() -> None:
    skills = build_skills()
    router = build_router(skills)

    def atender(entrada: str) -> None:
        nombre = router(entrada)
        print(f"[router -> skill: {nombre}]")
        print(skills[nombre].run(entrada))

    if len(sys.argv) > 1:
        entrada = leer_pregunta_o_argv("¿Que es RAG?")
        print(f"\n=== {entrada} ===")
        atender(entrada)
        return

    print("Asistente con skills. Escribi 'salir' para terminar.")
    print("Probá: '¿Que es ChromaDB?', 'Traducir: hola mundo', 'Cuanto es 8*9?'\n")
    while True:
        try:
            entrada = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if entrada.lower() in {"salir", "exit", "quit"}:
            break
        if entrada:
            atender(entrada)
            print()


if __name__ == "__main__":
    main()
