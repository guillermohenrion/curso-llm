"""
Servidor A2A minimo (protocolo Agent2Agent) - sin dependencias externas.

A2A (Agent2Agent, https://a2a-protocol.org) es un protocolo abierto para
que agentes construidos con frameworks o proveedores DISTINTOS se hablen
entre si por HTTP, sin conocer el codigo interno del otro. Dos piezas
centrales:

  1. Agent Card: un JSON en una URL fija (`/.well-known/agent-card.json`)
     que describe QUE sabe hacer el agente (sus "skills") y COMO hablarle
     (la URL de su endpoint). Es el "menu" que un cliente lee antes de
     mandar nada — ni siquiera necesita saber que framework usa el
     agente del otro lado.
  2. JSON-RPC 2.0 sobre HTTP: el cliente manda un Message con el metodo
     "message/send"; el servidor lo procesa como una Task y devuelve el
     resultado (status + artifacts).

Esta es una implementacion MINIMA y didactica (sin streaming, sin auth,
sin push notifications, sin manejo de estados intermedios como
"input-required") hecha solo con la libreria estandar de Python, para
ver el protocolo "en crudo" sin que un SDK lo esconda. Expone UN agente
traductor ES->EN de vocabulario chico, deliberadamente deterministico
(como el MockLLM de agent-orchestration-demo) — el foco del ejemplo es
el protocolo, no la inteligencia del agente.

Uso:
    python agente_servidor.py           # levanta en http://127.0.0.1:9000
    python cliente.py                   # (en otra terminal) le habla por A2A
"""
from __future__ import annotations

import json
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST, PORT = "127.0.0.1", 9000
AGENT_URL = f"http://{HOST}:{PORT}/"

# --- "Inteligencia" del agente: deliberadamente simple y deterministica ---
DICCIONARIO = {
    "hola": "hello", "mundo": "world", "gracias": "thank you",
    "buenos": "good", "dias": "days", "como": "how", "estas": "are you",
    "adios": "goodbye", "por": "for", "favor": "favor",
}


def traducir(texto: str) -> str:
    """Traduccion palabra por palabra contra un vocabulario chico de demo."""
    palabras = texto.lower().strip().split()
    traducidas = [DICCIONARIO.get(p.strip(".,?!"), f"[{p}]") for p in palabras]
    return " ".join(traducidas)


# --- Agent Card: lo primero que lee cualquier cliente A2A ---
AGENT_CARD = {
    "name": "traductor_a2a",
    "description": "Traduce frases simples de espanol a ingles (vocabulario de demo).",
    "url": AGENT_URL,
    "version": "0.1.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {
            "id": "traducir_es_en",
            "name": "Traducir espanol a ingles",
            "description": "Traduce palabra por palabra usando un vocabulario chico de demo.",
            "tags": ["traduccion", "demo"],
            "examples": ["hola mundo", "buenos dias"],
        }
    ],
}


class ManejadorA2A(BaseHTTPRequestHandler):
    def _responder_json(self, status: int, payload: dict) -> None:
        cuerpo = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, fmt, *args) -> None:  # silencia el log default, usamos el nuestro
        pass

    def do_GET(self) -> None:
        if self.path == "/.well-known/agent-card.json":
            print("[servidor] GET agent-card.json (descubrimiento)")
            self._responder_json(200, AGENT_CARD)
        else:
            self._responder_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        largo = int(self.headers.get("Content-Length", 0))
        cuerpo = self.rfile.read(largo)
        try:
            rpc = json.loads(cuerpo)
        except json.JSONDecodeError:
            self._responder_json(400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}})
            return

        metodo = rpc.get("method")
        if metodo != "message/send":
            self._responder_json(400, {
                "jsonrpc": "2.0", "id": rpc.get("id"),
                "error": {"code": -32601, "message": f"Metodo no soportado: {metodo}"},
            })
            return

        mensaje = rpc["params"]["message"]
        texto_entrada = next(
            (p["text"] for p in mensaje.get("parts", []) if p.get("kind") == "text"), ""
        )
        print(f'[servidor] message/send -> "{texto_entrada}"')

        respuesta = traducir(texto_entrada)
        task = {
            "id": str(uuid.uuid4()),
            "contextId": mensaje.get("contextId", str(uuid.uuid4())),
            "status": {"state": "completed"},
            "artifacts": [{
                "artifactId": str(uuid.uuid4()),
                "name": "traduccion",
                "parts": [{"kind": "text", "text": respuesta}],
            }],
        }
        print(f'[servidor] Task {task["id"]} completada -> "{respuesta}"')
        self._responder_json(200, {"jsonrpc": "2.0", "id": rpc.get("id"), "result": task})


def main() -> None:
    server = HTTPServer((HOST, PORT), ManejadorA2A)
    print(f"[servidor] Agent Card en http://{HOST}:{PORT}/.well-known/agent-card.json")
    print(f"[servidor] Escuchando JSON-RPC en {AGENT_URL} (Ctrl+C para salir)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
