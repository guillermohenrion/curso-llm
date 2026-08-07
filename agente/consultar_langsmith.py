"""
Consulta las ultimas corridas registradas en LangSmith para el proyecto del agente.

No modifica nada: solo lee (list_runs) usando las credenciales del .env y
muestra un resumen de cada corrida (nombre, estado, duracion, input/output).

Requisitos:
    - Mismo .env que usa agente_completo.py (LANGSMITH_API_KEY, LANGSMITH_PROJECT).
    - pip install langsmith python-dotenv  (ya estan en requirements-agente.txt)

Uso:
    python consultar_langsmith.py                # ultimas 10 corridas del proyecto
    python consultar_langsmith.py 20              # ultimas 20
    python consultar_langsmith.py 10 otro-proyecto  # otro proyecto puntual
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def main() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[error] No hay LANGSMITH_API_KEY en el .env. Copia .env.example a .env "
              "y completa tu API key.")
        sys.exit(1)

    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    proyecto = sys.argv[2] if len(sys.argv) > 2 else os.getenv("LANGSMITH_PROJECT", "curso-llm-agente")

    from langsmith import Client

    client = Client()
    print(f"Proyecto: {proyecto}  (mostrando hasta {limite} corridas, mas nuevas primero)\n")

    runs = list(client.list_runs(
        project_name=proyecto,
        run_type="chain",
        is_root=True,
        limit=limite,
    ))

    if not runs:
        print("No se encontraron corridas para ese proyecto. Verifica el nombre "
              "(LANGSMITH_PROJECT) o si el tracing estaba activo cuando corriste el agente.")
        return

    runs.sort(key=lambda r: r.start_time or 0, reverse=True)

    for r in runs:
        pregunta = (r.inputs or {}).get("input", "")
        respuesta = (r.outputs or {}).get("output", "") if r.outputs else "(sin salida / error)"
        duracion = ""
        if r.end_time and r.start_time:
            duracion = f"{(r.end_time - r.start_time).total_seconds():.1f}s"
        print(f"- [{r.status}] {r.start_time}  {duracion}")
        print(f"  Pregunta : {pregunta}")
        print(f"  Respuesta: {respuesta}")
        print(f"  URL      : https://smith.langchain.com/o/-/projects/p/{r.session_id}/r/{r.id}")
        print()


if __name__ == "__main__":
    main()
