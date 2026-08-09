# Agente completo (RAG + tools + memoria)

Un agente **conversacional** que integra, en un solo ejemplo, las piezas que en
el curso vimos por separado:

- **Tools**: el agente decide qué herramienta usar en cada paso.
  - `buscar_conocimiento` → RAG sobre el corpus del curso (Chroma + embeddings).
  - `calculadora` → aritmética con eval seguro.
  - `fecha_hoy` → fecha actual.
  - `guardar_nota` / `listar_notas` → un bloc de notas (estado que persiste entre turnos).
- **Razonamiento ReAct**: piensa → actúa → observa → repite, hasta responder.
- **Memoria**: recuerda el historial de la conversación (ventana de últimos turnos).

```
                 ┌───────────────────────────┐
   pregunta ───► │  AGENTE (ReAct, gemma3)    │
                 │  + historial de la charla  │
                 └────────────┬──────────────┘
        piensa / elige tool   │   observa el resultado y repite
        ┌─────────┬───────────┼───────────┬──────────────┐
        ▼         ▼           ▼           ▼              ▼
 buscar_conoc. calculadora fecha_hoy guardar_nota   listar_notas
        └─────────┴───────────┴───────────┴──────────────┘
                              ▼
                        Final Answer
```

## Requisitos

1. **Ollama** con los modelos:
   ```powershell
   ollama pull gemma3
   ollama pull nomic-embed-text
   ```
2. Dependencias de Python:
   ```powershell
   pip install -r requirements-agente.txt
   ```

## Correr

```powershell
# Modo interactivo (recomendado, para aprovechar la memoria)
python agente_completo.py

# Una sola pregunta
python agente_completo.py "¿Que es un agente? Y de paso, cuanto es 12*8?"
```

Ideas para probar en modo interactivo (y ver la memoria + las notas en acción):

```
> ¿Que es el patron ReAct?
> Recorda que la clase es el martes a las 18
> Cuanto es 15 * 7?
> ¿Que notas tengo guardadas?
```

**Lanzador `.ps1`** (verifica Ollama/modelos, activa el venv y corre el script):

```powershell
.\iniciar_agente_demo.ps1 "¿Que es un agente?"
```

## Observabilidad con LangSmith (opcional)

El agente puede registrar cada corrida en [LangSmith](https://smith.langchain.com),
donde se ve el **árbol ReAct completo**: cada `Thought → Action → Observation`, qué
tool se llamó, con qué input, latencias y tokens. Muy útil para depurar cuando el
modelo se desvía del formato.

1. Creá una cuenta gratis en https://smith.langchain.com y generá una **API key**
   (Settings → API Keys).
2. Copiá las credenciales a un `.env`:
   ```powershell
   copy .env.example .env
   ```
   Editá `.env` y completá `LANGSMITH_API_KEY`. El `.env` **no se sube al repo**.
3. Corré el agente normalmente: el tracing se activa solo si hay API key. Si no la
   configurás, el agente funciona igual pero sin registrar trazas.

Cada corrida se etiqueta con `run_name`, `tags` y `metadata` (el número de turno),
para poder filtrarlas en LangSmith.

### Consultar y reportar corridas (solo lectura)

Dos scripts leen las corridas registradas en LangSmith (no modifican nada):

```powershell
# Resumen en consola de las últimas corridas de un proyecto
python consultar_langsmith.py 10 curso-llm-rag_agente_react

# Reporte HTML analizando la ÚLTIMA corrida (ciclo ReAct paso a paso)
python reporte_langsmith.py curso-llm-rag_agente_react
```

`reporte_langsmith.py` reconstruye el árbol de la traza y genera
`reporte_langsmith.html` con: resumen (pregunta, respuesta, estado, duración,
tokens), análisis ReAct (iteraciones y tools usadas) y la línea de tiempo de cada
paso `llm` (Thought → Action) y `tool` (Observation). El HTML es un artefacto local
(está en `.gitignore`). Si no pasás el proyecto, usa `LANGSMITH_PROJECT` del `.env`.

## Cómo funciona

1. **Tools** — se arma la lista de herramientas (RAG + calculadora + fecha + notas).
   Cada tool tiene un `name` y una `description`; el agente usa esas descripciones
   para decidir cuál invocar.
2. **Agente ReAct** — `create_react_agent(llm, tools, prompt)` produce un agente que
   sigue el formato *Thought / Action / Action Input / Observation*. El `AgentExecutor`
   lo corre en bucle (con `max_iterations` y `handle_parsing_errors`).
3. **Memoria** — después de cada turno se guarda el par (pregunta, respuesta) y se
   inyecta el historial (últimos `MAX_TURNOS`) en el prompt, así el agente entiende
   seguimientos como "¿y eso para qué sirve?".
4. **Estado** — las notas viven en una lista en memoria; `guardar_nota` y `listar_notas`
   muestran cómo un agente puede acumular estado y consultarlo más tarde.

## Notas

- Usa **ReAct** (basado en prompt) para funcionar con `gemma3`, que no hace *tool
  calling* nativo. Con un modelo que soporte tools (p. ej. `llama3.1`) podés migrar
  a `create_tool_calling_agent` y evitar el parseo del formato ReAct.
- La memoria y las notas son **en memoria del proceso**: se reinician al cerrar. Para
  algo persistente habría que guardarlas en disco o en una base.
- El corpus de RAG está embebido en el script (mismo estilo que las otras demos). Para
  datos reales, indexá documentos propios y persistí el índice en disco.
- Este es el ejemplo más "de agente" del repo. Para orquestar **varios** agentes entre
  sí, ver [`../orquestacion/`](../orquestacion/README.md) (LangGraph).
