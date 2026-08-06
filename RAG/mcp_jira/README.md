# RAG + MCP de Jira

Parte del RAG simple y le suma acceso **en vivo a Jira** a través de un servidor
**MCP** (Model Context Protocol). Un agente decide, según la pregunta, si buscar
en la base de conocimiento del curso (tool de RAG) o consultar Jira (tools del MCP
de Atlassian: buscar issues con JQL, leer un ticket, etc.).

```
pregunta ──► AGENTE (LLM con tools)
                 ├─ buscar_conocimiento   → RAG local (Chroma + embeddings)
                 └─ tools de Jira (MCP)    → mcp-atlassian (subproceso via uvx)
```

## Requisitos

1. **Ollama** con un modelo que soporte *tool calling* (Gemma **no** sirve para esto):
   ```powershell
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```
2. **`uv`** instalado (provee `uvx`, que lanza el servidor MCP):
   https://docs.astral.sh/uv/getting-started/installation/
3. **Token de Jira (Cloud)**: https://id.atlassian.com/manage-profile/security/api-tokens
4. Dependencias de Python:
   ```powershell
   pip install -r requirements-mcp-jira.txt
   ```

## Configurar credenciales

```powershell
copy .env.example .env
```

Editá `.env` y completá `JIRA_URL`, `JIRA_USERNAME` y `JIRA_API_TOKEN`. El `.env`
**no se sube al repo**. Si no lo configurás, el agente arranca igual pero **solo**
con la tool de RAG (sin Jira).

## Correr

```powershell
# Consulta a Jira (usa las tools del MCP)
python rag_mcp_jira.py "Listá los issues abiertos del proyecto ABC"

# Pregunta conceptual (usa el RAG local)
python rag_mcp_jira.py "¿Que es una base de datos vectorial?"

# Interactivo
python rag_mcp_jira.py
```

## Cómo funciona

1. Se construye una **tool de RAG** (`buscar_conocimiento`) sobre un corpus del
   curso, indexado en Chroma con `OllamaEmbeddings`.
2. Se conecta al **MCP `mcp-atlassian`**, que se lanza como subproceso
   (`uvx mcp-atlassian`, transport `stdio`) con las credenciales de Jira. Sus
   herramientas se cargan como tools de LangChain vía `langchain-mcp-adapters`.
3. Se arma un **agente tool-calling** (`create_tool_calling_agent` + `AgentExecutor`)
   con `ChatOllama`. El LLM elige qué tool usar en cada paso.

## Seguridad: solo lectura

El acceso a Jira es **de solo lectura**, con doble protección:

1. **En el servidor MCP**: se lanza `mcp-atlassian` con `READ_ONLY_MODE=true`, que
   deshabilita todas las operaciones de escritura (crear/editar/borrar/comentar/
   transicionar).
2. **En el cliente**: además se **filtran** las tools cargadas y se descarta
   cualquiera cuyo nombre sugiera escritura (`create`, `update`, `delete`, `add`,
   `comment`, `transition`, etc.). Solo quedan tools de lectura (`get`, `search`,
   `list`, ...). El filtro es conservador: ante la duda, excluye.

Así el agente no puede modificar ni insertar nada en Jira, incluso si el modelo
lo intentara.

## Notas

- **Modelo con tools**: el ejemplo usa `llama3.1` por defecto (soporta tool calling
  en Ollama). Podés cambiarlo con `OLLAMA_TOOL_MODEL` en el `.env`. `gemma3` no
  hace tool calling nativo.
- El servidor `mcp-atlassian` se descarga/ejecuta solo la primera vez con `uvx`
  (no requiere instalación manual), pero **sí** requiere tener `uv`.
- El MCP de Atlassian también soporta Confluence y Jira Server/Data Center (con
  otras variables de entorno). Ver la documentación de `mcp-atlassian`.
- Este ejemplo hace llamadas reales a tu Jira: empezá con consultas de solo lectura.
