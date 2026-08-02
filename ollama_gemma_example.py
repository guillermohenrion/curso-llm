"""
Ejemplo minimo de uso de Ollama con el modelo Gemma.

Requisitos previos:
    1. Tener Ollama instalado y corriendo (https://ollama.com)
    2. Descargar el modelo:  ollama pull gemma3
    3. Instalar el cliente python:  pip install ollama
"""

import sys

import ollama

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "gemma3"  # cambia el tag segun lo que tengas descargado (ej: gemma3:1b, gemma2:9b)


def chat_simple(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def chat_streaming(prompt: str) -> None:
    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hola mundo"

    print(f"--- Respuesta simple ({pregunta!r}) ---")
    print(chat_simple(pregunta))

    print("\n--- Respuesta en streaming ---")
    chat_streaming(pregunta)
