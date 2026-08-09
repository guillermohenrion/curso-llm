"""
Orquestacion de agentes con Langflow (patron supervisor / router), version cliente.

A diferencia de orquestacion/multiagente_langgraph.py -donde el grafo de supervisor +
agentes especializados esta escrito en Python con LangGraph- aca el grafo se arma
VISUALMENTE en la UI de Langflow (ver README.md de esta carpeta para los pasos) y este
script solo lo INVOCA por su API REST. El "cerebro" de la orquestacion (que agente-tool
elige el supervisor) vive del lado de Langflow, no en este archivo.

Componentes (armados en la UI de Langflow, no en este script):
    - Un Agent 'supervisor' con tres Agent conectados como tools:
      matematico / traductor / explicador (cada uno con su propio Ollama + gemma3).

Requisitos:
    - Langflow corriendo:  langflow run   (ver README.md, requiere un venv propio)
    - Ollama con el modelo de chat:  ollama pull gemma3
    - El flujo ya armado y publicado en la UI, con su FLOW_ID y una API key.
    - pip install -r requirements-langflow.txt
    - copy .env.example .env   y completar LANGFLOW_FLOW_ID / LANGFLOW_API_KEY

Uso:
    python ejecutar_flow_langflow.py "Cuanto es 12 * (3 + 4)?"
    python ejecutar_flow_langflow.py "Traducir al ingles: hola mundo"
    python ejecutar_flow_langflow.py "Que es una base de datos vectorial?"
    python ejecutar_flow_langflow.py            # modo interactivo
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://127.0.0.1:7860").rstrip("/")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")
API_KEY = os.getenv("LANGFLOW_API_KEY")


def preguntar(pregunta: str) -> str:
    """Le manda la pregunta al flujo de Langflow via POST /api/v1/run/{FLOW_ID}."""
    if not FLOW_ID or not API_KEY:
        raise SystemExit(
            "Falta LANGFLOW_FLOW_ID o LANGFLOW_API_KEY en .env. "
            "Arma el flujo en la UI de Langflow y copia esos valores (ver README.md)."
        )

    resp = requests.post(
        f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}",
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        json={
            "input_value": pregunta,
            "input_type": "chat",
            "output_type": "chat",
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # La respuesta trae una lista de outputs por componente de salida del flujo;
    # tomamos el texto del ultimo mensaje del Chat Output.
    try:
        return data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except (KeyError, IndexError):
        return str(data)  # estructura inesperada: mostramos el JSON crudo para debug


def responder(pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    print("\n--- Respuesta final ---")
    print(preguntar(pregunta))


def main() -> None:
    argv = sys.argv[1:]
    if argv:
        responder(" ".join(argv))
        return

    print("Escribí una pregunta (o 'salir' para terminar):")
    while True:
        try:
            pregunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            responder(pregunta)


if __name__ == "__main__":
    main()
