# Orquestación de agentes con Langflow

Mismo patrón supervisor/router que [`orquestacion/`](../../orquestacion/) (LangGraph),
pero armado **visualmente** en Langflow en vez de en código, y disparado desde Python
vía su API REST.

```
pregunta ──► POST /api/v1/run/{FLOW_ID}  ──►  flujo de Langflow
                                                  │
                                          ┌───────┴────────┐
                                          │  Agent (super-  │  decide, según la
                                          │  visor), tools  │  pregunta, qué agente-
                                          │  = otros Agents │  tool invocar
                                          └───────┬────────┘
                              ┌───────────────────┼───────────────────┐
                              ▼                    ▼                   ▼
                        Agent: matematico    Agent: traductor    Agent: explicador
                        (Ollama, gemma3)      (Ollama, gemma3)    (Ollama, gemma3)
```

A diferencia de LangGraph (donde el ruteo es una función `add_conditional_edges` que
escribís a mano) o de Hermes (donde el ruteo lo hace el agente autónomo), acá el
"supervisor" es un componente **Agent** de Langflow al que le conectás **otros
componentes Agent como si fueran tools**: el LLM del supervisor lee la descripción de
cada tool-agente y decide cuál invocar.

## Requisitos

1. **Ollama** con el modelo de chat:
   ```powershell
   ollama pull gemma3
   ```
2. **Langflow** (entorno virtual propio recomendado, para no pisar las dependencias
   del resto del curso — Langflow fija versiones de `langchain`/`pydantic` distintas):
   ```powershell
   python -m venv .venv-langflow
   .venv-langflow\Scripts\activate
   pip install -r requirements-langflow.txt
   ```
3. Levantar Langflow:
   ```powershell
   langflow run
   ```
   Abre la UI en http://127.0.0.1:7860

## Armar el flujo (una vez, en la UI)

1. **Nuevo flujo en blanco.**
2. Arrastrá **tres componentes `Agent`**, uno por especialista. En cada uno:
   - Conectale como *Language Model* un componente **Ollama** (`Base URL:
     http://127.0.0.1:11434`, `Model Name: gemma3`).
   - Ponele `Name` y `Description` claros (la descripción es lo que el supervisor lee
     para decidir si lo usa):
     - `matematico` — "Resuelve cálculos y problemas aritméticos."
     - `traductor` — "Traduce texto entre idiomas."
     - `explicador` — "Explica conceptos generales o conceptuales."
   - En **Instructions/System Prompt** de cada uno, poné el mismo *system prompt* que
     usa el nodo equivalente en `orquestacion/multiagente_langgraph.py` (mismo texto,
     otra herramienta).
3. Arrastrá un **cuarto componente `Agent`** (el supervisor):
   - `Language Model`: otro componente **Ollama** (`gemma3`).
   - `Instructions`: *"Sos un supervisor. Según la pregunta del usuario, elegí y
     usá la tool-agente más apropiada: matematico, traductor o explicador."*
   - En el puerto **Tools** del supervisor, conectá los tres `Agent` del paso 2 (en
     Langflow, cualquier componente —incluyendo otro `Agent`— se puede conectar como
     tool de un `Agent`).
4. Conectá un componente **Chat Input** a la entrada del supervisor y un **Chat
   Output** a su salida.
5. Guardá el flujo y copiá su **Flow ID** (aparece en la URL del editor,
   `.../flow/<FLOW_ID>`).
6. En **Settings → API Keys**, generá una API key para llamar al flujo desde afuera.

## Configurar y correr el script

```powershell
copy .env.example .env
```

Completá `LANGFLOW_FLOW_ID` y `LANGFLOW_API_KEY` con los valores del paso anterior.

```powershell
python ejecutar_flow_langflow.py "Cuanto es 12 * (3 + 4)?"
python ejecutar_flow_langflow.py "Traducir al ingles: hola mundo"
python ejecutar_flow_langflow.py "Que es una base de datos vectorial?"
python ejecutar_flow_langflow.py                                        # interactivo
```

## Cómo funciona `ejecutar_flow_langflow.py`

El script **no** arma el flujo (eso se hace una sola vez, en la UI): sólo le manda la
pregunta a un flujo ya publicado, vía `POST /api/v1/run/{FLOW_ID}`, igual que lo haría
cualquier backend que quisiera usar Langflow como motor de orquestación. Es el mismo
rol que cumple `multiagente_langgraph.py`, pero ahí el grafo *es* el script; acá el
script es apenas un cliente HTTP del grafo (que vive en Langflow).

## Notas

- **Por qué un venv aparte**: Langflow fija versiones propias de `langchain-core` /
  `pydantic` que pueden chocar con las del resto del curso (mismo motivo por el que
  `orquestacion/requirements-orquestacion.txt` fija `langgraph<0.3`). Aislarlo evita
  romper los demás ejemplos.
- **Ventaja del patrón "Agent como tool de Agent"**: agregar un cuarto especialista es
  arrastrar un componente más y conectarlo al supervisor — no hay que tocar el resto
  del flujo, igual que en LangGraph agregar una rama nueva a `add_conditional_edges`.
- **Producción**: en vez de pegarle a `127.0.0.1:7860` a mano, Langflow se puede
  desplegar como servicio propio y el `LANGFLOW_URL` del `.env` apuntar a él.
