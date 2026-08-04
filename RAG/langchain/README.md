# RAG con LangChain — versiones incrementales

Cuatro ejemplos de RAG construidos con **LangChain + Ollama**, que van sumando
capacidades paso a paso. Todos comparten el mismo modelo local y el mismo corpus
(ver `comun.py`), así que la única diferencia entre versiones es la **técnica**.

| # | Archivo | Qué agrega |
|---|---------|-----------|
| 1 | `1_rag_basico.py`  | RAG mínimo: recuperar + generar (cadena LCEL). |
| 2 | `2_rag_memoria.py` | Conversación con **memoria avanzada**: retriever consciente del historial + ventana de mensajes con `trim_messages`. |
| 3 | `3_rag_tools.py`   | **Agente** ReAct con varias **tools**: RAG + calculadora + fecha. |
| 4 | `4_rag_skills.py`  | **Skills** de alto nivel + **router** que clasifica la intención y despacha. |
| 5 | `5_rag_langsmith.py` | **Observabilidad**: envía las trazas de cada ejecución a **LangSmith**. |

## Requisitos

1. **Ollama** corriendo (https://ollama.com) con los modelos:
   ```powershell
   ollama pull gemma3
   ollama pull nomic-embed-text
   ```
2. Dependencias de Python (con el `.venv` de la raíz activado):
   ```powershell
   pip install -r requirements-langchain.txt
   ```

## Cómo correr

Desde esta carpeta (`RAG/langchain/`):

```powershell
# 1. RAG básico
python 1_rag_basico.py "¿Que es una base de datos vectorial?"

# 2. RAG conversacional con memoria (modo interactivo)
python 2_rag_memoria.py
#   Probá una pregunta y después un seguimiento: "¿y eso para que sirve?"

# 3. Agente con tools
python 3_rag_tools.py "¿Que es ChromaDB? Y de paso, cuanto es 12*8?"

# 4. Skills + router
python 4_rag_skills.py "Traducir: los embeddings representan significado"
python 4_rag_skills.py "¿Que es LangChain?"

# 5. RAG con tracing en LangSmith (ver setup abajo)
python 5_rag_langsmith.py "¿Que es RAG?"
```

## LangSmith (ejemplo 5)

`5_rag_langsmith.py` registra cada ejecución en [LangSmith](https://smith.langchain.com)
para ver el árbol de pasos (retriever → prompt → LLM), latencias y tokens.

1. Creá una cuenta gratis en https://smith.langchain.com y generá una **API key**
   (Settings → API Keys).
2. Copiá las credenciales a un `.env`:
   ```powershell
   copy .env.example .env
   ```
   Editá `.env` y completá `LANGSMITH_API_KEY`. El `.env` **no se sube al repo**
   (está en `.gitignore`).
3. Corré el ejemplo: la traza aparece en tu proyecto de LangSmith. Si no configurás la
   API key, el RAG funciona igual pero sin registrar trazas.

## Progresión conceptual

1. **Básico** — el patrón esencial de RAG: `retriever → prompt → LLM`. Sin estado.
2. **Memoria** — para conversar. El retriever reformula la pregunta con el historial
   (así funcionan los seguimientos), y la memoria se acota a una ventana de mensajes
   para no crecer sin límite.
3. **Tools** — el LLM deja de seguir una cadena fija y pasa a ser un **agente** que
   decide qué herramienta usar en cada paso (RAG es solo una de las tools).
4. **Skills** — en vez de un bucle de razonamiento libre, un **router** clasifica la
   intención y despacha a una capacidad especializada. Más controlable y predecible
   que un agente abierto.

## Notas

- Los ejemplos usan `gemma3` (chat) y `nomic-embed-text` (embeddings) vía Ollama, igual
  que la demo `../rag_simple.py`.
- El vector store (ChromaDB) se construye **en memoria** cada vez, con el corpus de
  `comun.py`. Es suficiente para la clase; para datos reales conviene persistir en disco.
- El agente de tools usa **ReAct** (basado en prompt) para funcionar con Gemma sin
  necesidad de un modelo con *tool calling* nativo. Si usás un modelo que soporta
  herramientas (p. ej. `llama3.1`), podés migrar a `create_tool_calling_agent`.
