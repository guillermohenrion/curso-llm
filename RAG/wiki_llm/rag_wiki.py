"""
WIKI-LLM: RAG sobre una "wiki" de archivos markdown (patron de Andrej Karpathy).

En vez de una base vectorial con embeddings, la base de conocimiento es una
CARPETA DE ARCHIVOS MARKDOWN estructurados (titulo, Summary, Tags, contenido y
enlaces [[wiki]]). El LLM lee los archivos directamente; la "recuperacion" es
por archivos, no por chunks vectoriales.

Referencia del concepto:
    https://www.mindstudio.ai/blog/andrej-karpathy-llm-wiki-knowledge-base-claude-code

Dos modos de carga (como describe el articulo):
    - SELECTIVO (default): se hace un shortlist de las notas relevantes usando la
      linea 'Summary' y los 'Tags' (una especie de grep/fuzzy), y se cargan SOLO
      esas. Escala mejor en wikis grandes.
    - FULL (--full): se concatenan TODAS las notas. Simple y bueno para wikis
      chicas, porque el modelo puede conectar temas entre notas.

En ambos casos se antepone un system prompt de 'grounding' (responder solo desde
la wiki) y se cita de que archivos salio la respuesta.

Requisitos:
    - Ollama corriendo:  ollama pull gemma3
    - pip install -r requirements-wiki.txt

Uso:
    python rag_wiki.py "¿Que diferencia hay entre Chroma y Pinecone?"
    python rag_wiki.py --full "Resumi todo lo que hay sobre RAG"
    python rag_wiki.py            # modo interactivo
"""
from __future__ import annotations

import os
import re
import sys
import glob
import unicodedata

import ollama

sys.stdout.reconfigure(encoding="utf-8")

WIKI_DIR = os.path.join(os.path.dirname(__file__), "wiki")
CHAT_MODEL = "gemma3"
TOP_K = 3  # cuantas notas cargar en modo selectivo

SYSTEM_PROMPT = (
    "Sos un asistente de conocimiento. Abajo hay una wiki personal de notas.\n"
    "Al responder:\n"
    "- Respondé SOLO con el contenido de la wiki, salvo que se pida explicitamente "
    "conocimiento general.\n"
    "- Si la wiki no tiene informacion relevante, decilo explicitamente.\n"
    "- Citá las notas (por su nombre de archivo) de donde sacaste la respuesta.\n"
    "- Señalá si algo parece desactualizado o contradictorio."
)


def _norm(texto: str) -> str:
    """Minusculas y sin acentos, para comparar de forma robusta."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def cargar_notas() -> list[dict]:
    """Lee todas las notas .md y extrae metadata (titulo, summary, tags)."""
    notas = []
    for ruta in sorted(glob.glob(os.path.join(WIKI_DIR, "*.md"))):
        with open(ruta, encoding="utf-8") as f:
            raw = f.read()
        nombre = os.path.basename(ruta)
        titulo = next((l[2:].strip() for l in raw.splitlines() if l.startswith("# ")), nombre)
        summary = ""
        tags = ""
        for l in raw.splitlines():
            if l.lower().startswith("**summary**"):
                summary = l.split(":", 1)[-1].strip()
            elif l.lower().startswith("**tags**"):
                tags = l.split(":", 1)[-1].strip()
        notas.append({
            "archivo": nombre,
            "titulo": titulo,
            "summary": summary,
            "tags": tags,
            "raw": raw,
        })
    return notas


def shortlist(notas: list[dict], pregunta: str, k: int = TOP_K) -> list[dict]:
    """Puntua cada nota por coincidencia de palabras de la pregunta con
    titulo + summary + tags (el 'indice liviano' de la wiki)."""
    q_tokens = set(re.findall(r"\w+", _norm(pregunta)))
    # Palabras muy comunes que no aportan a la busqueda.
    stop = {"que", "de", "la", "el", "los", "las", "un", "una", "y", "o", "en",
            "es", "son", "hay", "entre", "para", "con", "del", "al", "como",
            "cual", "cuales", "sobre", "todo", "toda"}
    q_tokens -= stop

    puntuadas = []
    for n in notas:
        indice = _norm(f"{n['titulo']} {n['summary']} {n['tags']}")
        idx_tokens = set(re.findall(r"\w+", indice))
        score = len(q_tokens & idx_tokens)
        puntuadas.append((score, n))

    puntuadas.sort(key=lambda x: x[0], reverse=True)
    elegidas = [n for s, n in puntuadas if s > 0][:k]
    # Si nada matchea, caemos a cargar toda la wiki (es chica).
    return elegidas or notas


def construir_contexto(notas: list[dict]) -> str:
    """Concatena las notas, cada una prefijada con su nombre de archivo (para citar)."""
    partes = [f"\n\n# {n['archivo']}\n\n{n['raw']}" for n in notas]
    return "".join(partes)


def responder(notas: list[dict], pregunta: str, full: bool) -> None:
    """Elige que notas cargar (todas o shortlist) y le pide al LLM que responda con ellas."""
    seleccionadas = notas if full else shortlist(notas, pregunta)
    modo = "FULL (toda la wiki)" if full else "SELECTIVO (shortlist)"
    print(f"\n=== Pregunta: {pregunta} ===")
    print(f"[modo de carga: {modo}]")
    print(f"[notas cargadas: {', '.join(n['archivo'] for n in seleccionadas)}]\n")

    # Las notas van pegadas al system prompt (no como "documentos recuperados" separados):
    # asi el LLM las trata como su base de conocimiento, no como resultados de busqueda sueltos.
    contexto = construir_contexto(seleccionadas)
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n---\n" + contexto + "\n---"},
        {"role": "user", "content": pregunta},
    ]
    resp = ollama.chat(model=CHAT_MODEL, messages=mensajes)
    print(resp["message"]["content"])


def main() -> None:
    argv = sys.argv[1:]
    full = "--full" in argv  # flag de modo, no una pregunta
    argv = [a for a in argv if a != "--full"]  # el resto de los argumentos SI es la pregunta

    notas = cargar_notas()
    if not notas:
        print(f"No encontre notas .md en {WIKI_DIR}")
        sys.exit(1)
    print(f"Wiki cargada: {len(notas)} notas en {WIKI_DIR}")

    if argv:
        responder(notas, " ".join(argv), full)
    else:
        print("Escribí una pregunta (o 'salir' para terminar). Tip: --full no aplica en interactivo.\n")
        while True:
            try:
                pregunta = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if pregunta:
                responder(notas, pregunta, full)


if __name__ == "__main__":
    main()
