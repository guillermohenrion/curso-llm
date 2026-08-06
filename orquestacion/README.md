# Orquestacion de agentes con LangGraph

Ejemplo de **orquestacion de agentes**: en lugar de un unico agente que hace
todo, hay varios agentes especializados y un **supervisor** que decide, segun
la pregunta, a quien derivarla. Se modela con **LangGraph** como un grafo de
estados.

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
                       END
```

## Requisitos

1. **Ollama** con el modelo de chat:
   ```powershell
   ollama pull gemma3
   ```
2. Dependencias de Python:
   ```powershell
   pip install -r requirements-orquestacion.txt
   ```

## Correr

```powershell
# El supervisor deriva al agente matematico
python multiagente_langgraph.py "Cuanto es 12 * (3 + 4)?"

# Deriva al traductor
python multiagente_langgraph.py "Traducir al ingles: hola mundo"

# Deriva al explicador
python multiagente_langgraph.py "Que es una base de datos vectorial?"

# Modo interactivo
python multiagente_langgraph.py
```

**Lanzador `.ps1`** (verifica Ollama/modelo, activa el venv y corre el script):

```powershell
.\iniciar_multiagente_demo.ps1 "Cuanto es 12 * 8?"
```

## Como funciona

1. **Estado compartido** (`Estado`, un `TypedDict`): un diccionario con `pregunta`,
   `ruta` y `respuesta` que todos los nodos leen y actualizan.
2. **Nodos** del grafo (cada uno es una funcion):
   - `supervisor`: le pide al LLM que clasifique la pregunta en una ruta
     (`matematico` / `traductor` / `explicador`) y la guarda en el estado.
   - `matematico`, `traductor`, `explicador`: los agentes especializados, cada
     uno con su propio *system prompt*. El `matematico` ademas usa una
     mini-herramienta (eval aritmetico seguro), para mostrar que cada agente
     puede tener sus propias capacidades.
3. **Aristas**:
   - `START -> supervisor`
   - `supervisor -> (agente elegido)` con `add_conditional_edges`: el ruteo es
     **dinamico**, lo decide la funcion `_elegir_ruta` a partir del estado.
   - `(agente) -> END`
4. Se compila el grafo con `.compile()` y se ejecuta con `.invoke(...)`.

## Notas

- **LangGraph** es la libreria de LangChain para orquestar flujos con estado,
  ciclos y ramificaciones. Un LLM suelto responde una vez; con LangGraph podes
  encadenar decisiones, derivar entre agentes y (si hiciera falta) volver atras.
- Este ejemplo usa un ruteo de **un salto** (supervisor -> agente -> fin) para
  que se entienda el patron. A partir de aca se puede extender: agentes que se
  llaman entre si, un nodo revisor que valida y reenvia, memoria/checkpoints, etc.
- **Version de LangGraph**: se fija `langgraph<0.3` a proposito. LangGraph 0.3+
  (y 1.x) exige `langchain-core` 1.x, que rompe el `langchain` 0.3.x que usa el
  resto del curso.
- `gemma3` alcanza para el supervisor y los agentes de texto porque solo se le
  pide clasificar y generar; no se usa *tool calling* nativo.
