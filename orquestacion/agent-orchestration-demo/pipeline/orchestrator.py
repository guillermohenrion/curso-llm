"""
PipelineOrchestrator
=====================
Ejecuta la cadena A1 -> A2 -> A3 de forma explícita y muestra en consola,
paso a paso, todo lo que ocurre: qué prompt recibe cada agente, qué
información recibe como input y qué resultado produce.

El flujo es intencionalmente lineal y "leído de arriba hacia abajo":

    result_1 = A1.run(request)
    result_2 = A2.run(request, result_1)
    result_3 = A3.run(request, result_1, result_2)
    return result_3

Cada agente solo conoce lo que el orquestador decide pasarle. El
orquestador es quien conoce el ORDEN del pipeline (A1 -> A2 -> A3); los
agentes no saben nada sobre quién viene antes o después de ellos.
"""

from common.llm_mock import MockLLM
from pipeline.agents import ContentAgent, EditorAgent, TitleAgent

SEPARATOR = "=" * 40
SUBSEPARATOR = "-" * 40


def _print_header(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def _print_step(title: str) -> None:
    print(f"\n{SUBSEPARATOR}\n{title}\n{SUBSEPARATOR}")


def _print_block(label: str, content) -> None:
    print(f"\n{label}:")
    print(content)


class PipelineOrchestrator:
    """Orquesta el pipeline secuencial de generación de documentos."""

    def __init__(self, llm=None):
        # Las tres etapas comparten el mismo LLM (mock). En un escenario
        # real cada agente podría incluso usar un modelo distinto.
        llm = llm or MockLLM()
        self.title_agent = TitleAgent(llm)
        self.content_agent = ContentAgent(llm)
        self.editor_agent = EditorAgent(llm)

    def run(self, user_request: str) -> str:
        _print_header("PIPELINE ORCHESTRATION")
        _print_block("USER REQUEST", user_request)

        # --- A1: TITLE AGENT ------------------------------------------------
        _print_step("A1 - TITLE AGENT")
        result_1 = self.title_agent.run(user_request)
        _print_block("PROMPT", self.title_agent.last_prompt)
        _print_block("INPUT", user_request)
        _print_block("OUTPUT", result_1)

        # --- A2: CONTENT AGENT -----------------------------------------------
        _print_step("A2 - CONTENT AGENT")
        result_2 = self.content_agent.run(user_request, result_1)
        _print_block("PROMPT", self.content_agent.last_prompt)
        _print_block("INPUT", {"user_request": user_request, "titles": result_1})
        _print_block("OUTPUT", result_2)

        # --- A3: EDITOR AGENT --------------------------------------------------
        _print_step("A3 - EDITOR AGENT")
        result_3 = self.editor_agent.run(user_request, result_1, result_2)
        _print_block("PROMPT", self.editor_agent.last_prompt)
        _print_block("INPUT", {"titles": result_1, "content": result_2})
        _print_block("OUTPUT", result_3)

        _print_header("FINAL RESULT")
        print(result_3)

        return result_3
