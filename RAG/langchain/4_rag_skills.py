"""
RAG con SKILLS cargadas desde archivos markdown (patron "Agent Skills").

Una "skill" es una capacidad definida en su PROPIO archivo .md (carpeta skills/),
con un header (frontmatter YAML) que declara QUE hace, y un cuerpo con las
instrucciones para el LLM. El agente NO tiene las skills hardcodeadas: las
descubre leyendo la carpeta.

Idea clave -> carga a demanda (progressive disclosure):
    - Para DECIDIR, el router lee SOLO los headers (name + description) de cada
      skill. Es barato y no mete el prompt entero de cada skill en el contexto.
    - Recien cuando se elige una skill se carga su CUERPO completo (las
      instrucciones) y se ejecuta.

Formato de un archivo de skill (skills/<nombre>.md):
    ---
    name: traducir
    description: Traducir al ingles un texto que provee el usuario.
    ---
    Sos un traductor. Traduci al ingles el texto del usuario...

Campos opcionales del header:
    - retrieval: true          -> la skill inyecta contexto de la base vectorial (RAG)
    - handler: <nombre_python>  -> la skill la resuelve una funcion Python (no el LLM)

Flujo:
    pregunta -> ROUTER (lee headers, elige skill) -> se carga el cuerpo -> respuesta

Uso:
    python 4_rag_skills.py
    python 4_rag_skills.py "Traducir: los embeddings representan significado"
    python 4_rag_skills.py "¿Que es LangChain?"
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from comun import get_llm, get_retriever, format_docs, leer_pregunta_o_argv

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


# ---------------------------------------------------------------------------
# Modelo de una skill
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    nombre: str                       # del header: name
    descripcion: str                  # del header: description  (lo unico que ve el router)
    ruta: str                         # path al .md (el cuerpo se carga a demanda)
    meta: dict = field(default_factory=dict)  # resto del header (retrieval, handler, ...)
    cuerpo: str | None = None         # instrucciones; None hasta que se cargan


# ---------------------------------------------------------------------------
# Parser minimo de frontmatter (sin dependencias extra)
# ---------------------------------------------------------------------------
def _parse_frontmatter(texto: str) -> tuple[dict, str]:
    """Separa el header '--- ... ---' del cuerpo. Devuelve (dict_header, cuerpo).

    split("---", 2) corta como maximo 2 veces, asi que un '---' que aparezca
    despues (p. ej. un separador dentro del cuerpo de la skill) no rompe el
    parseo: partes queda ['', header, resto-del-cuerpo-con-sus-propios-'---'].
    No es YAML real (no soporta listas/anidamiento) a proposito: alcanza para
    'name'/'description'/'retrieval'/'handler' y evita sumar una dependencia.
    """
    if not texto.startswith("---"):
        return {}, texto.strip()
    partes = texto.split("---", 2)  # ['', header, cuerpo]
    if len(partes) < 3:
        return {}, texto.strip()
    header: dict = {}
    for linea in partes[1].strip().splitlines():
        if ":" in linea:
            clave, _, valor = linea.partition(":")
            header[clave.strip()] = valor.strip()
    return header, partes[2].strip()


def _leer_header(ruta: str) -> dict:
    """Lee SOLO el header de una skill (carga barata para el router)."""
    with open(ruta, encoding="utf-8") as f:
        header, _ = _parse_frontmatter(f.read())
    return header


def cargar_catalogo() -> dict[str, Skill]:
    """Descubre las skills de skills/*.md leyendo solo sus headers (no el cuerpo)."""
    catalogo: dict[str, Skill] = {}
    for archivo in sorted(os.listdir(SKILLS_DIR)):
        if not archivo.endswith(".md"):
            continue
        ruta = os.path.join(SKILLS_DIR, archivo)
        header = _leer_header(ruta)
        nombre = header.get("name") or os.path.splitext(archivo)[0]
        catalogo[nombre] = Skill(
            nombre=nombre,
            descripcion=header.get("description", ""),
            ruta=ruta,
            meta={k: v for k, v in header.items() if k not in ("name", "description")},
        )
    return catalogo


def cargar_cuerpo(skill: Skill) -> str:
    """Carga (a demanda) el cuerpo/instrucciones de la skill y lo cachea."""
    if skill.cuerpo is None:
        with open(skill.ruta, encoding="utf-8") as f:
            _, cuerpo = _parse_frontmatter(f.read())
        skill.cuerpo = cuerpo
    return skill.cuerpo


# ---------------------------------------------------------------------------
# Handlers de codigo (para skills con 'handler: <nombre>' en el header)
# ---------------------------------------------------------------------------
def _eval_aritmetico(entrada: str, skill: Skill) -> str:
    permitido = set("0123456789+-*/(). ")
    expr = "".join(ch for ch in entrada if ch in permitido).strip()
    if not expr:
        return "No encontre una expresion aritmetica en la entrada."
    try:
        return f"Resultado: {eval(expr, {'__builtins__': {}}, {})}"  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return f"Error al evaluar '{expr}': {e}"


HANDLERS = {
    "eval_aritmetico": _eval_aritmetico,
}


# ---------------------------------------------------------------------------
# Ejecucion de una skill (segun su header)
# ---------------------------------------------------------------------------
def ejecutar_skill(skill: Skill, entrada: str) -> str:
    """Ejecuta la skill elegida: por handler de codigo, con RAG, o como prompt simple."""
    cuerpo = cargar_cuerpo(skill)

    # 1) Skill respaldada por codigo (handler Python).
    handler = skill.meta.get("handler")
    if handler:
        fn = HANDLERS.get(handler)
        if fn is None:
            return f"[error] La skill '{skill.nombre}' declara un handler desconocido: {handler}"
        return fn(entrada, skill)

    llm = get_llm()

    # 2) Skill con recuperacion (RAG): el cuerpo es la instruccion + inyectamos contexto.
    if skill.meta.get("retrieval", "").lower() == "true":
        retriever = get_retriever(k=2)
        prompt = ChatPromptTemplate.from_template(
            cuerpo + "\n\nContexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
        )
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return chain.invoke(entrada)

    # 3) Skill de prompt simple: el cuerpo es el system prompt.
    prompt = ChatPromptTemplate.from_messages([("system", cuerpo), ("human", "{entrada}")])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"entrada": entrada})


# ---------------------------------------------------------------------------
# Router: elige la skill leyendo SOLO los headers (name + description)
# ---------------------------------------------------------------------------
def build_router(catalogo: dict[str, Skill]):
    """Devuelve una funcion que, dado un texto, elige que skill deberia atenderlo."""
    catalogo_txt = "\n".join(f"- {s.nombre}: {s.descripcion}" for s in catalogo.values())
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Elegí la skill mas adecuada para atender la peticion del usuario.\n"
        "Skills disponibles:\n{catalogo}\n\n"
        "Peticion: {entrada}\n\n"
        "Respondé UNICAMENTE con el nombre de la skill (una sola palabra), sin explicar."
    )
    chain = prompt | llm | StrOutputParser()

    def router(entrada: str) -> str:
        # El LLM deberia responder solo el nombre, pero a veces agrega texto
        # alrededor (p. ej. "La skill 'traducir'."); por eso no comparamos
        # igualdad exacta, sino que buscamos el nombre de cada skill DENTRO
        # de lo que respondio.
        elegido = chain.invoke({"catalogo": catalogo_txt, "entrada": entrada}).strip().lower()
        for nombre in catalogo:
            if nombre in elegido:
                return nombre
        # Si no matcheo ninguna (respuesta rara / vacia), no fallamos: caemos
        # a 'rag' como default razonable, o a la primera skill si ni esa existe.
        return "rag" if "rag" in catalogo else next(iter(catalogo))  # fallback

    return router


def main() -> None:
    catalogo = cargar_catalogo()
    print(f"[skills descubiertas en skills/]: {', '.join(catalogo)}\n")
    router = build_router(catalogo)

    def atender(entrada: str) -> None:
        nombre = router(entrada)
        print(f"[router -> skill: {nombre}]  (se carga el cuerpo de {os.path.basename(catalogo[nombre].ruta)})")
        print(ejecutar_skill(catalogo[nombre], entrada))

    if len(sys.argv) > 1:
        entrada = leer_pregunta_o_argv("¿Que es RAG?")
        print(f"=== {entrada} ===")
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
