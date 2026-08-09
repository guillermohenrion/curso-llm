# Patrones de orquestación de agentes

El curso ya tiene un ejemplo de orquestación con **código** ([`orquestacion/`](../orquestacion/),
LangGraph). Esta carpeta suma el **mismo patrón conceptual** (un supervisor que deriva una
pregunta a agentes especializados) resuelto con otras dos herramientas, para comparar
**tres formas distintas de construir lo mismo**:

| Ejemplo | Cómo se define la orquestación | Dónde vive el "supervisor" |
|---|---|---|
| [`orquestacion/`](../orquestacion/) | Código Python (grafo de estados) | Un nodo del grafo, con `add_conditional_edges` |
| [`langflow/`](langflow/) | Flujo visual (drag & drop) + API REST | Un componente **Agent** con otros agentes conectados como *tools* |
| [`hermes/`](hermes/) | Agente autónomo (CLI) que aprende *skills* con el tiempo | El propio Hermes, delegando en sub-agentes vía MCP |

## El patrón común

```
                 ┌──────────────┐
   pregunta ───► │  supervisor  │  clasifica y elige la ruta
                 └──────┬───────┘
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  ┌───────────┐   ┌───────────┐   ┌─────────────┐
  │ matematico│   │ traductor │   │  explicador │   agentes especializados
  └─────┬─────┘   └─────┬─────┘   └──────┬──────┘
        └───────────────┴────────────────┘
                        ▼
                     respuesta
```

Un único agente que "hace de todo" es difícil de mantener y de dar *prompts* precisos.
Separarlo en agentes chicos y especializados, con un supervisor que rutea, es más fácil
de razonar, de testear y de extender (agregar un agente nuevo no toca a los demás).

## Por qué dos herramientas tan distintas

- **LangGraph** (`orquestacion/`): control total, todo en código, versionable con git.
  Requiere saber programar el grafo a mano.
- **Langflow** (`langflow/`): mismo patrón pero armado **visualmente**, pensado para
  iterar rápido o para que alguien no-programador pueda ajustar el flujo. El flujo se
  dispara luego desde código vía su API REST.
- **Hermes** (`hermes/`): no es una librería que uno importa, es un **agente autónomo**
  (CLI) que ya trae su propio bucle de orquestación, memoria y capacidad de spawnear
  sub-agentes; uno le conecta *tools* propias vía **MCP** (el mismo protocolo que ya
  usamos en [`RAG/mcp_jira/`](../RAG/mcp_jira/)) en vez de programar el grafo.

## Requisito común

Los tres ejemplos usan **Ollama** en local:

```powershell
ollama pull gemma3
```

Cada subcarpeta tiene su propio `README.md` con los pasos específicos.
