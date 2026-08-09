"""
Punto de entrada del ejemplo de PIPELINE.

Ejecutar desde la carpeta agent-orchestration-demo/:

    python pipeline/main.py
"""

import sys
from pathlib import Path

# Asegura que los acentos se impriman bien en consolas Windows (cp1252).
sys.stdout.reconfigure(encoding="utf-8")

# Permite ejecutar este archivo directamente (python pipeline/main.py)
# agregando la raíz del proyecto a sys.path para que `common` y
# `pipeline` se puedan importar como paquetes.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import PipelineOrchestrator  # noqa: E402


def main() -> None:
    user_request = "Crear un documento sobre Inteligencia Artificial aplicada a banca."
    orchestrator = PipelineOrchestrator()
    orchestrator.run(user_request)


if __name__ == "__main__":
    main()
