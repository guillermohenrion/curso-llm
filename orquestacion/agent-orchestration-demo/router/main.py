"""
Punto de entrada del ejemplo de ROUTER.

Ejecutar desde la carpeta agent-orchestration-demo/:

    python router/main.py

Ejecuta 4 casos:

1. Solicitud sobre préstamos          -> credit_agent
2. Solicitud sobre inversiones        -> investment_agent
3. Solicitud que mezcla ambos temas   -> credit_agent + investment_agent (en paralelo)
4. Solicitud sobre tarjetas           -> card_agent
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Asegura que los acentos se impriman bien en consolas Windows (cp1252).
sys.stdout.reconfigure(encoding="utf-8")

# Permite ejecutar este archivo directamente (python router/main.py)
# agregando la raíz del proyecto a sys.path para que `common` y
# `router` se puedan importar como paquetes.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_mock import MockLLM  # noqa: E402
from common.models import AgentResult  # noqa: E402
from router.aggregator import Aggregator  # noqa: E402
from router.agents import CardAgent, CreditAgent, InvestmentAgent  # noqa: E402
from router.router import RouterAgent  # noqa: E402

SEPARATOR = "=" * 40


def print_tag(tag: str, content=None) -> None:
    print(f"\n[{tag}]")
    if content is not None:
        print(content)


def build_agent_map(llm) -> dict:
    return {
        "credit_agent": CreditAgent(llm),
        "investment_agent": InvestmentAgent(llm),
        "card_agent": CardAgent(llm),
    }


def run_case(title: str, user_request: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")

    print_tag("USER", user_request)

    llm = MockLLM()
    router = RouterAgent()

    print_tag(
        "ROUTER",
        f"Catálogo disponible: {[a['name'] for a in router.registry]}",
    )

    decision = router.route(user_request)
    print_tag(
        "ROUTING DECISION",
        f"selected_agents={decision.selected_agents}\nreason={decision.reason}",
    )

    if not decision.selected_agents:
        print_tag("FINAL RESPONSE", "No hay ningún agente disponible para esta solicitud.")
        return

    agent_map = build_agent_map(llm)
    selected = [agent_map[name] for name in decision.selected_agents]

    for agent in selected:
        print_tag("AGENT START", f"{agent.name} -> procesando solicitud...")

    results: list[AgentResult]
    if len(selected) > 1:
        # Ejecución en paralelo: el Router dispara varios agentes
        # especializados al mismo tiempo con ThreadPoolExecutor.
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = [executor.submit(agent.run, user_request) for agent in selected]
            results = [future.result() for future in futures]
    else:
        results = [selected[0].run(user_request)]

    for result in results:
        print_tag(
            "AGENT RESULT",
            f"agent={result.agent} status={result.status} "
            f"execution_time_ms={result.metadata['execution_time_ms']}\n"
            f"answer={result.answer}",
        )

    aggregator = Aggregator()
    final_response = aggregator.aggregate(user_request, results)

    print_tag(
        "AGGREGATOR",
        f"Combinando {len(results)} resultado(s) de: {final_response.agents_used}",
    )
    print_tag(
        "FINAL RESPONSE",
        f"query={final_response.query}\n"
        f"agents_used={final_response.agents_used}\n\n"
        f"{final_response.answer}",
    )


def main() -> None:
    run_case("CASO 1 - Prestamo  =>  credit_agent", "Quiero pedir un préstamo.")
    run_case("CASO 2 - Inversion  =>  investment_agent", "Quiero invertir en bonos.")
    run_case(
        "CASO 3 - Prestamo + Inversion  =>  credit_agent + investment_agent",
        "Quiero comparar un préstamo con una inversión.",
    )
    run_case("CASO 4 - Tarjeta  =>  card_agent", "Quiero aumentar el límite de mi tarjeta.")


if __name__ == "__main__":
    main()
