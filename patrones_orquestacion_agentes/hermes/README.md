# Orquestación de agentes con Hermes

[Hermes](https://hermes-ai.net/es/) (Nous Research) es distinto a LangGraph o Langflow:
**no es una librería que uno importa** para armar un grafo propio, es un **agente
autónomo** que ya trae su propio orquestador (delega en sub-agentes, aprende *skills*
de la experiencia, corre en CLI o conectado a Telegram/Discord/Slack/etc.). No tiene
SDK de Python: la forma soportada de sumarle capacidades propias es exponerlas como un
**servidor MCP** (el mismo protocolo que ya usamos en
[`RAG/mcp_jira/`](../../RAG/mcp_jira/), pero ahí lo *consumíamos*; acá lo *proveemos*).

```
pregunta ──►  Hermes (CLI, orquestador propio)
                  │
                  │  decide que necesita resolver un calculo/traduccion/concepto
                  ▼
          MCP 'agentes-locales'  (mcp_server_agentes.py, este ejemplo)
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   resolver_   traducir_  explicar_
   calculo     texto      concepto
   (Ollama)    (Ollama)   (Ollama)
```

En vez de programar el ruteo (como en LangGraph) o armarlo visualmente (como en
Langflow), acá el ruteo lo hace **Hermes solo**: ve las tres tools del servidor MCP,
lee sus descripciones y elige cuál invocar según la tarea. El patrón sigue siendo el
mismo (supervisor + especialistas), pero el supervisor no lo escribimos nosotros.

## Requisitos

1. **Ollama** con el modelo de chat:
   ```powershell
   ollama pull gemma3
   ```
2. Dependencias de Python para el servidor MCP:
   ```powershell
   pip install -r requirements-hermes.txt
   ```
3. **Hermes** instalado (ver instrucciones oficiales, resumidas abajo).

## Instalar Hermes

```powershell
# Windows (PowerShell)
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

El instalador gestiona `uv`, Python y Node.js por su cuenta. Después:

```powershell
hermes setup      # asistente de configuracion (proveedor de modelo, etc.)
```

> Hermes usa su propio LLM (podés apuntarlo a un proveedor cloud o, si lo soporta tu
> version, a Ollama). Los tres agentes de este ejemplo corren con **Ollama local**
> independientemente del modelo que use Hermes para orquestar.

## Conectar este servidor MCP a Hermes

Los servidores MCP se registran con un `command`/`args`/`env`, igual que ya viste en
`RAG/mcp_jira/rag_mcp_jira.py` (que lanza `uvx mcp-atlassian`). Acá el servidor lo
escribimos nosotros y lo lanzamos con Python:

```powershell
hermes tools
```

Y agregá una entrada apuntando a este script (la ruta exacta de "agregar servidor MCP"
puede variar entre versiones de Hermes; revisá `hermes tools --help` o la
[documentación de Hermes](https://hermes-ai.net/es/) si difiere):

```json
{
  "mcpServers": {
    "agentes-locales": {
      "command": "python",
      "args": ["mcp_server_agentes.py"],
      "cwd": "RUTA/A/patrones_orquestacion_agentes/hermes"
    }
  }
}
```

## Probar el servidor MCP solo (sin Hermes)

Antes de conectarlo a Hermes conviene verificar que el servidor arranca y expone las
tools, con el inspector oficial de MCP:

```powershell
mcp dev mcp_server_agentes.py
```

Abre una UI donde podés invocar `resolver_calculo`, `traducir_texto` y
`explicar_concepto` a mano y ver la respuesta, sin depender de Hermes todavía.

## Usar

Con el servidor registrado, hablale a Hermes normalmente (CLI o el gateway de
mensajería que hayas configurado):

```
> Cuanto es 12 * (3 + 4)?         -> Hermes elige la tool resolver_calculo
> Traduci al ingles: hola mundo   -> Hermes elige la tool traducir_texto
> Que es una base de datos vectorial?  -> Hermes elige la tool explicar_concepto
```

## Cómo funciona `mcp_server_agentes.py`

- Usa **FastMCP** (del SDK oficial `mcp`) para exponer tres funciones Python como
  tools MCP, vía `@mcp.tool()`. Cada tool tiene un *docstring* claro: es lo único que
  Hermes ve para decidir cuál usar, así que hace las veces de "descripción del
  especialista" (igual rol que el `Description` de cada `Agent` en el ejemplo de
  Langflow, o el *system prompt* de cada nodo en LangGraph).
- Cada tool arma su propio `ChatOllama(model="gemma3")` con un *system prompt*
  especializado (matemático / traductor / explicador) — el mismo contenido que en
  `orquestacion/multiagente_langgraph.py`, solo que expuesto como tool en vez de nodo
  de grafo.
- El transporte es **stdio**: Hermes lanza este script como subproceso y le habla por
  stdin/stdout con el protocolo MCP, igual que `uvx mcp-atlassian` en el ejemplo de
  Jira (pero ahí el subproceso lo escribió otro equipo; acá lo escribimos nosotros).

## Notas

- **Por qué MCP y no un SDK de Python**: Hermes no expone (a la fecha de este ejemplo)
  una API de Python para embeberlo en un script propio; su superficie programática
  soportada es CLI + gateways de mensajería + MCP. Si más adelante Hermes suma un SDK,
  este mismo servidor MCP debería seguir funcionando igual (es un componente
  independiente, no atado a como lo invoquen).
- **Reutilizar el patrón de seguridad de `RAG/mcp_jira/`**: si en vez de tools locales
  quisieras exponerle a Hermes algo con efectos reales (una base de datos, un ticketing),
  aplicá el mismo criterio de "solo lectura por defecto" que ahí: filtrá o deshabilitá
  explícitamente cualquier tool de escritura hasta confiar en el flujo.
- **gemma3 alcanza** porque las tools NO dependen de que gemma3 haga tool-calling: el
  tool-calling lo hace Hermes (contra las tools MCP); gemma3 solo genera texto adentro
  de cada tool.
