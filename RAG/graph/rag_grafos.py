"""
GRAPH RAG: RAG basado en un grafo de conocimiento.

En vez de recuperar "chunks" de texto de una base vectorial, aca la base de
conocimiento es un GRAFO: nodos = conceptos, aristas = relaciones (tripletas
sujeto -> relacion -> objeto). Ante una pregunta:

    1. Se detectan las entidades relevantes del grafo (por similitud de embedding
       entre la pregunta y el nombre de cada nodo).
    2. Se recupera el vecindario de esos nodos (las tripletas a 1-2 saltos).
    3. Ese subgrafo, serializado como texto, es el contexto que recibe el LLM.

Ventaja del enfoque en grafo: el contexto son relaciones explicitas y
conectadas, no fragmentos sueltos. Util para preguntas que requieren "conectar"
varios hechos.

Requisitos:
    - Ollama corriendo con:  ollama pull gemma3 ; ollama pull nomic-embed-text
    - pip install -r requirements-graph.txt

Uso:
    python rag_grafos.py
    python rag_grafos.py "¿Que diferencia hay entre Chroma y Pinecone?"
"""
from __future__ import annotations

import sys

import networkx as nx
import numpy as np
import ollama

sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3"
TOP_NODOS = 3       # cuantas entidades matchear con la pregunta
SALTOS = 2          # profundidad del vecindario a recuperar
UMBRAL = 0.45       # similitud minima para considerar un nodo relevante

# --- Grafo de conocimiento (tripletas: sujeto, relacion, objeto) ---
TRIPLETAS = [
    ("RAG", "combina", "recuperacion de informacion"),
    ("RAG", "combina", "generacion de texto"),
    ("RAG", "usa", "base de datos vectorial"),
    ("base de datos vectorial", "almacena", "embeddings"),
    ("embeddings", "representan", "significado del texto"),
    ("base de datos vectorial", "busca por", "similitud coseno"),
    ("Chroma", "es una", "base de datos vectorial"),
    ("Pinecone", "es una", "base de datos vectorial"),
    ("Chroma", "corre", "local"),
    ("Pinecone", "corre", "en la nube"),
    ("Ollama", "corre", "modelos locales"),
    ("Ollama", "soporta", "Gemma"),
    ("Ollama", "soporta", "nomic-embed-text"),
    ("nomic-embed-text", "genera", "embeddings"),
    ("Gemma", "es un", "LLM"),
    ("LLM", "genera", "texto"),
    ("LangChain", "es un", "framework"),
    ("LangChain", "ofrece", "retrievers"),
    ("LangChain", "ofrece", "agentes"),
    ("LangChain", "ofrece", "memoria"),
    ("agentes", "usan", "tools"),
    ("RAG", "mejora", "LLM"),
]


def build_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for s, r, o in TRIPLETAS:
        g.add_edge(s, o, relacion=r)
    return g


def embed(texto: str) -> np.ndarray:
    v = ollama.embeddings(model=EMBED_MODEL, prompt=texto)["embedding"]
    return np.array(v, dtype=float)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def matchear_entidades(g: nx.MultiDiGraph, pregunta: str) -> list[str]:
    """Devuelve los nodos mas similares a la pregunta (por embedding)."""
    q = embed(pregunta)
    nodos = list(g.nodes)
    sims = [(_cos(q, embed(n)), n) for n in nodos]
    sims.sort(reverse=True)
    elegidos = [n for s, n in sims[:TOP_NODOS] if s >= UMBRAL]
    # Si nada supera el umbral, al menos devolvemos el mejor match.
    return elegidos or [sims[0][1]]


def subgrafo_contexto(g: nx.MultiDiGraph, entidades: list[str]) -> list[str]:
    """Recupera las tripletas del vecindario (SALTOS) de las entidades dadas."""
    ug = g.to_undirected()
    cercanos: set[str] = set()
    for e in entidades:
        if e in ug:
            cercanos |= set(nx.single_source_shortest_path_length(ug, e, cutoff=SALTOS).keys())

    hechos: list[str] = []
    for s, o, data in g.edges(data=True):
        if s in cercanos or o in cercanos:
            hechos.append(f"{s} --{data['relacion']}--> {o}")
    return sorted(set(hechos))


def generar_respuesta(pregunta: str, hechos: list[str]) -> str:
    contexto = "\n".join(f"- {h}" for h in hechos)
    prompt = f"""Sos un asistente que responde usando SOLO los hechos del grafo de
conocimiento (relaciones sujeto -> relacion -> objeto). Si no alcanzan, decilo.

Hechos del grafo:
{contexto}

Pregunta: {pregunta}

Respuesta:"""
    resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def rag(g: nx.MultiDiGraph, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    entidades = matchear_entidades(g, pregunta)
    print(f"\n--- Entidades detectadas: {', '.join(entidades)}")
    hechos = subgrafo_contexto(g, entidades)
    print("--- Subgrafo recuperado (hechos) ---")
    for h in hechos:
        print(f"  {h}")
    print("\n--- Respuesta del modelo ---")
    print(generar_respuesta(pregunta, hechos))


def main() -> None:
    g = build_graph()
    print(f"Grafo de conocimiento: {g.number_of_nodes()} nodos, "
          f"{g.number_of_edges()} relaciones.")

    if len(sys.argv) > 1:
        rag(g, " ".join(sys.argv[1:]))
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
                rag(g, pregunta)


if __name__ == "__main__":
    main()
