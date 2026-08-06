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
