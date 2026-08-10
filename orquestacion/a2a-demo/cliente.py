"""
Cliente A2A minimo - habla con agente_servidor.py usando SOLO la libreria
estandar (urllib), para ver el protocolo sin que un SDK lo esconda.

Dos pasos, que son el corazon de A2A:
  1. Descubrimiento: pedir el Agent Card (que sabe hacer el agente, y a
     que URL hablarle). El cliente no necesita importar ningun codigo
     del servidor — solo HTTP + JSON.
  2. Invocacion: mandar un mensaje via JSON-RPC (metodo "message/send")
     y leer la Task que devuelve.

Uso (con agente_servidor.py corriendo en otra terminal):
    python cliente.py "hola mundo gracias"
    python cliente.py                # usa un mensaje de ejemplo
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid

SERVIDOR = "http://127.0.0.1:9000"


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def descubrir_agente() -> dict:
    """Paso 1: lee el Agent Card, el 'menu' publico del agente."""
    url = f"{SERVIDOR}/.well-known/agent-card.json"
    with urllib.request.urlopen(url, timeout=5) as resp:
        card = json.loads(resp.read())
    print(f"[cliente] Agent Card descubierta: {card['name']} - {card['description']}")
    for skill in card["skills"]:
        print(f"[cliente]   skill: {skill['id']} ({skill['description']})")
    return card


def enviar_mensaje(card: dict, texto: str) -> str:
    """Paso 2: manda el texto como Task via JSON-RPC message/send."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": texto}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    print(f'[cliente] Enviando Task -> "{texto}"')
    respuesta = _post_json(card["url"], payload)

    if "error" in respuesta:
        raise RuntimeError(f"Error A2A: {respuesta['error']}")

    task = respuesta["result"]
    print(f"[cliente] Task {task['id']} -> status={task['status']['state']}")
    return task["artifacts"][0]["parts"][0]["text"]


def main() -> None:
    texto = " ".join(sys.argv[1:]) or "hola mundo gracias"
    try:
        card = descubrir_agente()
    except urllib.error.URLError:
        print("[error] No se pudo conectar al servidor A2A. "
              "Corre primero: python agente_servidor.py")
        sys.exit(1)

    resultado = enviar_mensaje(card, texto)
    print(f"\n--- Resultado ---\n{resultado}")


if __name__ == "__main__":
    main()
