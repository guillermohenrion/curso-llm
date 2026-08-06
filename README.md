# Curso LLM - ITBA

Material práctico del curso de **Large Language Models (LLMs)**. Este README
explica cómo instalar y correr cada ejemplo; para la teoría detrás de cada
técnica (tokenización, embeddings, attention, RAG, chunking, etc.) y qué hace
cada librería, ver [`TEORIA.md`](TEORIA.md).

Este repositorio contiene dos bloques independientes:

1. **Clase 1 - Notebook de fundamentos** (`clase1_llm_practica.ipynb`): tokenización,
   embeddings, visualización y mecanismo de atención. No necesita GPU y corre tanto
   local como en Google Colab.
2. **Demos locales con Ollama** (`ollama_gemma_example.py`, `RAG/rag_simple.py`): chat con
   el modelo Gemma y un RAG simple 100% local (en la carpeta `RAG/`). Requieren tener
   Ollama instalado.

Además hay tres notebooks de verificación con modelos de Hugging Face:
- `prueba_modelo_huggingface.ipynb` — corre `distilgpt2` y `distilbert` usando la caché
  por defecto de Hugging Face. Prueba de humo tras instalar las dependencias.
- `prueba_modelo_local_hf.ipynb` — **descarga** `google/flan-t5-small` a la carpeta
  `./models/` y lo ejecuta desde disco con `local_files_only=True` (útil para correr sin
  internet). La carpeta `models/` está en `.gitignore`, así que los pesos no se suben al repo.
- `prueba_modelo_hf_remoto.ipynb` — llama a `gpt2` y `distilbert-base-uncased` **sin
  descargarlos**, vía la Inference API de Hugging Face (`InferenceClient`). Requiere un
  token gratis de HF — ver [Sección 5](#5-demo-remota-con-la-inference-api-de-hugging-face-opcional).

Y en la carpeta `RAG/` hay varias variantes de RAG, de menor a mayor complejidad:
- `RAG/rag_simple.py` — RAG local sin frameworks (Ollama + ChromaDB). Ver [Sección 4](#4-demos-locales-con-ollama-opcional).
- `RAG/langchain/` — 5 versiones incrementales con **LangChain** (básico, memoria, tools,
  skills, LangSmith). Ver [Sección 6](#6-rag-con-langchain-opcional).
- `RAG/pinecone/` — RAG simple con **Pinecone** (base vectorial en la nube). Ver [Sección 7](#7-rag-con-pinecone-opcional).
- `RAG/graph/` — **Graph RAG**: recuperación sobre un grafo de conocimiento. Ver [Sección 8](#8-graph-rag-opcional).
- `RAG/wiki_llm/` — **Wiki-LLM**: RAG sobre una wiki de markdown (patrón de Karpathy). Ver [Sección 9](#9-wiki-llm-opcional).
- `RAG/pdf/` — RAG sobre una **carpeta de PDFs** (indexa en ChromaDB y consulta). Ver [Sección 10](#10-rag-sobre-pdfs-opcional).
- `RAG/mcp_jira/` — RAG + **MCP de Jira**: agente que combina el RAG con acceso de **solo lectura** a Jira. Ver [Sección 11](#11-rag--mcp-de-jira-opcional).

Y por fuera del RAG, hay ejemplos de **agentes**:
- `agente/` — un **agente completo** (RAG + tools + memoria) con LangChain. Ver [Sección 12](#12-agente-completo-opcional).
- `orquestacion/` — varios agentes especializados coordinados por un supervisor, con **LangGraph**. Ver [Sección 13](#13-orquestación-de-agentes-con-langgraph-opcional).

## Guía rápida (TL;DR)

```powershell
# 1. Entorno (una sola vez)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Notebook de fundamentos (Clase 1)  -> abrir clase1_llm_practica.ipynb en VS Code

# 3. Cualquier demo con Ollama necesita los modelos:
ollama pull gemma3
ollama pull nomic-embed-text

# 4. Elegí qué correr:
python RAG\rag_simple.py "¿Que es RAG?"                    # RAG local (Chroma)
python RAG\langchain\1_rag_basico.py "¿Que es RAG?"        # RAG con LangChain
python RAG\pinecone\rag_pinecone.py "¿Que es RAG?"         # RAG con Pinecone (necesita .env)
```

Cada sección de abajo explica los detalles y los requisitos de cada bloque.

---

## 1. Requisitos previos

- **Python 3.11** (recomendado). Verificá tu versión con:
  ```powershell
  python --version
  ```
- **Git** (para clonar el repo) — opcional si ya tenés la carpeta.
- Conexión a internet la primera vez (se descargan modelos de Hugging Face y GloVe).
- Para las demos de Ollama: **[Ollama](https://ollama.com)** instalado.

> Nota Windows: los comandos de abajo son para **PowerShell**. Si usás CMD o
> macOS/Linux, activá el entorno virtual con el comando equivalente (ver más abajo).

---

## 2. Puesta en marcha del notebook (Clase 1)

### Paso 1 — Crear el entorno virtual

Parate en la carpeta del proyecto (`curso-llm`) y creá el venv una sola vez:

```powershell
python -m venv .venv
```

### Paso 2 — Activar el entorno virtual

```powershell
# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1

# CMD (Windows)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Cuando está activo vas a ver `(.venv)` al comienzo de la línea de la terminal.

> Si PowerShell bloquea la activación con un error de "execution policy", corré una vez:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### Paso 3 — Instalar las dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala todo lo necesario para el notebook (torch, transformers, gensim, etc.).
La descarga pesa varios cientos de MB y puede tardar unos minutos la primera vez.

### Paso 4 — Abrir el notebook

Tenés dos opciones:

- **VS Code**: abrí `clase1_llm_practica.ipynb` y seleccioná el kernel `.venv`
  (arriba a la derecha, "Select Kernel" → Python del entorno `.venv`).
- **Jupyter en el navegador**:
  ```powershell
  pip install jupyter
  jupyter notebook clase1_llm_practica.ipynb
  ```

Corré las celdas en orden. La primera celda de instalación (`!pip install ...`) está
pensada para Colab; en local ya instalaste todo con `requirements.txt`, así que podés
saltearla.

---

## 3. Contenido del notebook

El notebook recorre toda la cadena de procesamiento de un LLM moderno:

1. **Tokenización comparada** — BPE (GPT-2) vs WordPiece (BERT) vs multilingüe, y cómo
   se mide la "fragmentación" (tokens por palabra) en español e inglés.
2. **Embeddings clásicos con Word2Vec** — entrenamiento de un modelo de juguete y uso de
   vectores pre-entrenados GloVe para explorar similitudes y analogías.
3. **Visualización de embeddings** — reducción a 2D con t-SNE y UMAP.
4. **Self-Attention y Multi-Head Attention desde cero** — implementación en PyTorch.
5. **Atención en modelos reales** — comparación de DistilBERT (bidireccional) vs GPT-2
   (causal).

Cada sección tiene una explicación breve, código comentado y un ejercicio para completar.

---

## 4. Demos locales con Ollama (opcional)

Estas demos corren un LLM en tu propia máquina, sin API en la nube.

### Paso 1 — Instalar Ollama y descargar los modelos

Instalá Ollama desde https://ollama.com y luego descargá los modelos:

```powershell
ollama pull gemma3            # modelo generador (chat)
ollama pull nomic-embed-text  # modelo de embeddings (solo para el RAG)
```

### Paso 2 — Instalar las dependencias de Python

Con el venv activado:

```powershell
# Demo de Gemma
pip install ollama

# Demo de RAG (deps propias en la carpeta RAG/)
pip install -r RAG\requirements-rag.txt
```

### Paso 3 — Correr las demos

Los scripts `.ps1` verifican que Ollama esté corriendo, descargan los modelos si
faltan, activan el venv y ejecutan el ejemplo:

```powershell
# Chat simple con Gemma
.\iniciar_gemma_demo.ps1 "¿Qué es un LLM?"

# RAG simple (modo interactivo)
.\RAG\iniciar_rag_demo.ps1

# RAG con una pregunta puntual
.\RAG\iniciar_rag_demo.ps1 "¿Qué es RAG?"
```

También podés correr los scripts de Python directamente (con el venv activo y Ollama corriendo):

```powershell
python ollama_gemma_example.py "hola mundo"
python RAG\rag_simple.py "¿Qué es una base de datos vectorial?"
```

---

## 5. Demo remota con la Inference API de Hugging Face (opcional)

`prueba_modelo_hf_remoto.ipynb` llama modelos de Hugging Face **sin descargarlos**: la
inferencia corre en los servidores de HF (vía `InferenceClient`, provider `hf-inference`,
que es gratis y no pide tarjeta). Sirve para comparar con el enfoque local de
`prueba_modelo_huggingface.ipynb`.

### Paso 1 — Conseguir un token gratis

1. Creá una cuenta en [huggingface.co](https://huggingface.co) si no tenés.
2. Generá un token en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (permiso **Read** alcanza).

### Paso 2 — Guardar el token en un `.env` local

```powershell
copy .env.example .env
```

Editá `.env` y completá `HF_TOKEN=hf_...` con tu token real. **Este archivo nunca se sube
al repo** (está en `.gitignore`) — si no lo creás, el notebook te lo va a pedir de forma
segura con `getpass` al correrlo.

### Paso 3 — Instalar dependencias y correr

```powershell
pip install -r requirements-hf-remoto.txt
```

Abrí `prueba_modelo_hf_remoto.ipynb` y corré las celdas en orden.

> Nota: la Inference API gratuita (`hf-inference`) ya no sirve modelos de generación de
> texto libre (como `gpt2`) — los movieron a proveedores de pago aparte. Por eso el
> notebook usa **question answering** y **fill-mask**, que sí están disponibles gratis.

---

## 6. RAG con LangChain (opcional)

Cinco ejemplos incrementales en `RAG/langchain/`, todos sobre Ollama
(`gemma3` + `nomic-embed-text`). Detalle completo en
[`RAG/langchain/README.md`](RAG/langchain/README.md).

### Paso 1 — Modelos de Ollama (si no los bajaste aún)

```powershell
ollama pull gemma3
ollama pull nomic-embed-text
```

### Paso 2 — Instalar dependencias

```powershell
pip install -r RAG\langchain\requirements-langchain.txt
```

### Paso 3 — Correr (en orden de complejidad creciente)

```powershell
cd RAG\langchain

python 1_rag_basico.py "¿Que es una base de datos vectorial?"   # RAG minimo (LCEL)
python 2_rag_memoria.py                                          # conversacional con memoria
python 3_rag_tools.py "¿Que es ChromaDB? Y cuanto es 12*8?"      # agente ReAct con tools
python 4_rag_skills.py "Traducir: hola mundo"                    # skills + router
python 5_rag_langsmith.py "¿Que es RAG?"                         # con tracing en LangSmith

cd ..\..
```

Para el ejemplo 5 (LangSmith) necesitás un token: copiá `RAG\langchain\.env.example`
a `RAG\langchain\.env` y completá `LANGSMITH_API_KEY`. Sin token el RAG funciona igual,
pero no registra trazas.

**Lanzadores `.ps1`** (verifican Ollama/modelos, activan el venv y corren el script):

```powershell
.\RAG\langchain\iniciar_1_basico.ps1 "¿Que es RAG?"
.\RAG\langchain\iniciar_2_memoria.ps1
.\RAG\langchain\iniciar_3_tools.ps1 "¿Que es ChromaDB?"
.\RAG\langchain\iniciar_4_skills.ps1 "Traducir: hola mundo"
.\RAG\langchain\iniciar_5_langsmith.ps1 "¿Que es RAG?"
```

---

## 7. RAG con Pinecone (opcional)

RAG simple usando **Pinecone** (base vectorial en la nube) en vez de ChromaDB.
Embeddings y generación siguen locales con Ollama. Detalle completo en
[`RAG/pinecone/README.md`](RAG/pinecone/README.md).

### Paso 1 — Token de Pinecone

Creá una cuenta gratis en [pinecone.io](https://www.pinecone.io), generá una API key
(https://app.pinecone.io → API Keys) y guardala en un `.env`:

```powershell
copy RAG\pinecone\.env.example RAG\pinecone\.env
#  editá el .env y pegá tu PINECONE_API_KEY
```

### Paso 2 — Instalar dependencias y correr

```powershell
pip install -r RAG\pinecone\requirements-pinecone.txt

# (con los modelos de Ollama ya descargados)
python RAG\pinecone\rag_pinecone.py "¿Que es una base de datos vectorial?"
```

El script crea el índice solo si no existe (dimensión detectada automáticamente,
métrica coseno, serverless).

**Lanzador `.ps1`** (corta con un aviso claro si falta el `.env` con `PINECONE_API_KEY`):

```powershell
.\RAG\pinecone\iniciar_pinecone_demo.ps1 "¿Que es una base de datos vectorial?"
```

---

## 8. Graph RAG (opcional)

RAG sobre un **grafo de conocimiento** (nodos = conceptos, aristas = relaciones)
en `RAG/graph/`. Detalle en [`RAG/graph/README.md`](RAG/graph/README.md).

```powershell
ollama pull gemma3
ollama pull nomic-embed-text
pip install -r RAG\graph\requirements-graph.txt

python RAG\graph\rag_grafos.py "¿Que diferencia hay entre Chroma y Pinecone?"
```

Detecta las entidades relevantes por embeddings, recupera su vecindario (tripletas
a 1-2 saltos) y le pasa ese subgrafo al LLM como contexto.

**Lanzador `.ps1`**:

```powershell
.\RAG\graph\iniciar_graph_demo.ps1 "¿Que diferencia hay entre Chroma y Pinecone?"
```

---

## 9. Wiki-LLM (opcional)

RAG sobre una **wiki de archivos markdown** (patrón de Andrej Karpathy): el LLM lee
los archivos directamente, sin embeddings ni base vectorial. En `RAG/wiki_llm/`.
Detalle en [`RAG/wiki_llm/README.md`](RAG/wiki_llm/README.md).

```powershell
ollama pull gemma3
pip install -r RAG\wiki_llm\requirements-wiki.txt

# selectivo (shortlist por Summary/Tags)
python RAG\wiki_llm\rag_wiki.py "¿Que diferencia hay entre Chroma y Pinecone?"
# full (carga toda la wiki)
python RAG\wiki_llm\rag_wiki.py --full "Resumi todo lo que hay sobre RAG"
```

La base de conocimiento son los `.md` de `RAG/wiki_llm/wiki/`; agregar notas nuevas
no requiere reindexar nada.

**Lanzador `.ps1`**:

```powershell
.\RAG\wiki_llm\iniciar_wiki_demo.ps1 "¿Que diferencia hay entre Chroma y Pinecone?"
```

---

## 10. RAG sobre PDFs (opcional)

Indexa una **carpeta de PDFs** en ChromaDB y permite consultarlos, citando archivo
y página. En `RAG/pdf/`. Detalle en [`RAG/pdf/README.md`](RAG/pdf/README.md).

```powershell
ollama pull gemma3
ollama pull nomic-embed-text
pip install -r RAG\pdf\requirements-pdf.txt

# 1) poné tus PDFs en RAG\pdf\docs\
# 2) indexá (incremental) y preguntá:
python RAG\pdf\rag_pdf.py "¿De que trata el documento?"
python RAG\pdf\rag_pdf.py --docs C:\ruta\a\pdfs     # otra carpeta
python RAG\pdf\rag_pdf.py --reindex                 # reconstruir indice
```

Extrae el texto (`pypdf`), lo parte en chunks, los embebe con `nomic-embed-text` y
los guarda en un índice persistente (`chroma_pdf/`). El indexado es incremental.

Hay además una variante con **LangChain + LangSmith** (observabilidad/tracing):

```powershell
pip install -r RAG\pdf\requirements-pdf-langsmith.txt
copy RAG\pdf\.env.example RAG\pdf\.env    # opcional: completar LANGSMITH_API_KEY
python RAG\pdf\rag_pdf_langsmith.py "¿De que trata el documento?"
```

Cada consulta queda registrada en [LangSmith](https://smith.langchain.com) (árbol
retriever → prompt → LLM, latencias, tokens). Sin token funciona igual, sin trazas.

Y una variante que cambia el **método de chunking**: en vez de cortar cada N
caracteres con solapamiento (`rag_pdf.py`), agrupa **oraciones completas** hasta
un tamaño objetivo, sin cortar nunca una oración al medio:

```powershell
python RAG\pdf\rag_pdf_semantico.py "¿De que trata el documento?"
```

Usa su propio índice (`chroma_pdf_semantico/`) para no mezclarse con los otros
dos. Buen contraste para mostrar en clase: mismo PDF, mismo `TOP_K`, pero
fragmentos más coherentes (y de tamaño más variable) al no partir oraciones.

**Lanzadores `.ps1`** (avisan si `docs/` está vacía):

```powershell
.\RAG\pdf\iniciar_pdf_demo.ps1 "¿De que trata el documento?"
.\RAG\pdf\iniciar_pdf_langsmith_demo.ps1 "¿De que trata el documento?"
.\RAG\pdf\iniciar_pdf_semantico_demo.ps1 "¿De que trata el documento?"
```

---

## 11. RAG + MCP de Jira (opcional)

Un agente que combina el RAG del curso con acceso **en vivo a Jira** vía un
servidor **MCP** (Model Context Protocol). El acceso a Jira es de **solo lectura**
(doble protección: `READ_ONLY_MODE` en el servidor + filtro de tools en el cliente).
En `RAG/mcp_jira/`. Detalle en [`RAG/mcp_jira/README.md`](RAG/mcp_jira/README.md).

```powershell
# Modelo que soporte tool-calling (gemma3 NO sirve para esto):
ollama pull llama3.1
ollama pull nomic-embed-text

# uv instalado (provee uvx, que lanza el servidor MCP): https://docs.astral.sh/uv/
pip install -r RAG\mcp_jira\requirements-mcp-jira.txt

# Token de Jira Cloud en .env (si no lo configurás, el agente arranca solo con el RAG):
copy RAG\mcp_jira\.env.example RAG\mcp_jira\.env
#  editá el .env y completá JIRA_URL / JIRA_USERNAME / JIRA_API_TOKEN

python RAG\mcp_jira\rag_mcp_jira.py "Listá los issues abiertos del proyecto ABC"  # usa Jira
python RAG\mcp_jira\rag_mcp_jira.py "¿Que es una base de datos vectorial?"        # usa el RAG
```

El agente decide en cada pregunta si buscar en la base de conocimiento del curso
(tool de RAG local) o consultar Jira (tools del MCP `mcp-atlassian`). No puede
modificar ni insertar nada en Jira.

---

## 12. Agente completo (opcional)

Un agente **conversacional** que integra en un solo ejemplo las piezas del curso:
**tools** (RAG + calculadora + fecha + bloc de notas), **razonamiento ReAct** y
**memoria** de la conversación. En `agente/`. Detalle en
[`agente/README.md`](agente/README.md).

```powershell
ollama pull gemma3
ollama pull nomic-embed-text
pip install -r agente\requirements-agente.txt

# Modo interactivo (recomendado, para aprovechar la memoria y las notas)
python agente\agente_completo.py

# Una sola pregunta
python agente\agente_completo.py "¿Que es un agente? Y de paso, cuanto es 12*8?"
```

El agente decide en cada paso qué herramienta usar, recuerda el hilo de la charla
y puede acumular estado (las notas). Es el ejemplo más "de agente" del repo.

Además tiene **observabilidad opcional con LangSmith**: si copiás `agente\.env.example`
a `agente\.env` y completás `LANGSMITH_API_KEY`, cada corrida registra el árbol ReAct
completo (Thought → Action → Observation). Sin token funciona igual, sin trazas.

**Lanzador `.ps1`**:

```powershell
.\agente\iniciar_agente_demo.ps1 "¿Que es un agente?"
```

---

## 13. Orquestación de agentes con LangGraph (opcional)

En lugar de un único agente que hace todo, hay varios agentes especializados y
un **supervisor** que decide, según la pregunta, a quién derivarla. Se modela
con **LangGraph** como un grafo de estados. En `orquestacion/`. Detalle en
[`orquestacion/README.md`](orquestacion/README.md).

```powershell
ollama pull gemma3
pip install -r orquestacion\requirements-orquestacion.txt

python orquestacion\multiagente_langgraph.py "Cuanto es 12 * (3 + 4)?"        # -> matematico
python orquestacion\multiagente_langgraph.py "Traducir al ingles: hola mundo" # -> traductor
python orquestacion\multiagente_langgraph.py "Que es una base de datos vectorial?" # -> explicador
```

El supervisor clasifica la pregunta y una arista condicional la deriva al agente
correspondiente (`matematico` / `traductor` / `explicador`). El agente matemático
usa además una mini-herramienta propia (eval aritmético seguro).

**Lanzador `.ps1`**:

```powershell
.\orquestacion\iniciar_multiagente_demo.ps1 "Cuanto es 12 * 8?"
```

---

## 14. Estructura del repositorio

```
curso-llm/
├── clase1_llm_practica.ipynb          # Notebook principal (Clase 1), listo para repartir a alumnos
├── prueba_modelo_huggingface.ipynb    # Notebook de prueba: corre un modelo open de HF (cache)
├── prueba_modelo_local_hf.ipynb       # Descarga un modelo de HF a ./models/ y lo corre offline
├── prueba_modelo_hf_remoto.ipynb      # Llama modelos de HF via Inference API (sin descargar)
├── ollama_gemma_example.py            # Demo de chat con Gemma (Ollama)
├── iniciar_gemma_demo.ps1             # Lanzador de la demo de Gemma
├── RAG/                               # Demos de RAG
│   ├── rag_simple.py                  #   RAG local sin frameworks (Ollama + ChromaDB)
│   ├── iniciar_rag_demo.ps1           #   lanzador de la demo de RAG
│   ├── requirements-rag.txt           #   dependencias del RAG simple
│   ├── langchain/                     #   RAG con LangChain (versiones incrementales)
│   │   ├── comun.py                   #     modulo compartido (modelos, corpus, retriever)
│   │   ├── 1_rag_basico.py            #     RAG minimo (LCEL)
│   │   ├── iniciar_1_basico.ps1       #     lanzador
│   │   ├── 2_rag_memoria.py           #     RAG conversacional con memoria avanzada
│   │   ├── iniciar_2_memoria.ps1      #     lanzador
│   │   ├── 3_rag_tools.py             #     agente ReAct con tools
│   │   ├── iniciar_3_tools.ps1        #     lanzador
│   │   ├── 4_rag_skills.py            #     skills desde .md (carga a demanda) + router
│   │   ├── skills/                    #       una skill por archivo .md (header + cuerpo)
│   │   ├── iniciar_4_skills.ps1       #     lanzador
│   │   ├── 5_rag_langsmith.py         #     RAG con tracing en LangSmith
│   │   ├── iniciar_5_langsmith.ps1    #     lanzador (avisa si falta .env)
│   │   └── requirements-langchain.txt #     dependencias de LangChain
│   ├── pinecone/                      #   RAG simple con Pinecone (base vectorial en la nube)
│   │   ├── rag_pinecone.py            #     script principal (lee token de .env)
│   │   ├── iniciar_pinecone_demo.ps1  #     lanzador (corta si falta .env)
│   │   └── requirements-pinecone.txt  #     dependencias de Pinecone
│   ├── graph/                         #   Graph RAG (grafo de conocimiento)
│   │   ├── rag_grafos.py              #     match de entidades + vecindario + LLM
│   │   ├── iniciar_graph_demo.ps1     #     lanzador
│   │   └── requirements-graph.txt     #     dependencias (networkx, numpy, ollama)
│   ├── wiki_llm/                      #   Wiki-LLM (RAG sobre markdown, patron Karpathy)
│   │   ├── rag_wiki.py                #     lee la wiki, shortlist y consulta al LLM
│   │   ├── iniciar_wiki_demo.ps1      #     lanzador
│   │   ├── requirements-wiki.txt      #     dependencias (solo ollama)
│   │   └── wiki/                      #     base de conocimiento (.md)
│   ├── pdf/                           #   RAG sobre una carpeta de PDFs
│   │   ├── rag_pdf.py                 #     version simple, chunking por caracteres+overlap
│   │   ├── iniciar_pdf_demo.ps1       #     lanzador (avisa si docs/ esta vacia)
│   │   ├── rag_pdf_semantico.py       #     misma version, chunking por oraciones completas
│   │   ├── iniciar_pdf_semantico_demo.ps1 #  lanzador
│   │   ├── rag_pdf_langsmith.py       #     version LangChain + tracing en LangSmith
│   │   ├── iniciar_pdf_langsmith_demo.ps1 #  lanzador
│   │   ├── requirements-pdf.txt       #     dependencias de las versiones simples
│   │   ├── requirements-pdf-langsmith.txt #  dependencias de la version LangSmith
│   │   └── docs/                      #     poné aca tus PDFs (no se versionan)
│   └── mcp_jira/                      #   RAG + MCP de Jira (solo lectura)
│       ├── rag_mcp_jira.py            #     agente: tool de RAG + tools de Jira (MCP)
│       ├── .env.example               #     plantilla de credenciales de Jira
│       ├── README.md                  #     detalle del ejemplo
│       └── requirements-mcp-jira.txt  #     dependencias (langchain-mcp-adapters, etc.)
├── agente/                            # Agente completo (RAG + tools + memoria)
│   ├── agente_completo.py             #   agente ReAct conversacional con 5 tools
│   ├── iniciar_agente_demo.ps1        #   lanzador
│   ├── .env.example                   #   credenciales de LangSmith (opcional)
│   ├── requirements-agente.txt        #   dependencias
│   └── README.md                      #   detalle del ejemplo
├── orquestacion/                      # Orquestacion de agentes (no RAG)
│   ├── multiagente_langgraph.py       #   supervisor + agentes especializados (LangGraph)
│   ├── iniciar_multiagente_demo.ps1   #   lanzador
│   ├── requirements-orquestacion.txt  #   dependencias (langgraph, etc.)
│   └── README.md                      #   detalle del ejemplo
├── requirements.txt                   # Dependencias del notebook
├── requirements-hf-remoto.txt         # Dependencias de la demo remota de HF
├── .env.example                       # Plantilla para el token de HF (copiar a .env)
├── TEORIA.md                          # Teoria detras de cada ejemplo + glosario de librerias
└── README.md                          # Este archivo
```

---

## 15. Problemas frecuentes

- **`SSLCertVerificationError` al descargar modelos** (redes con proxy corporativo):
  `requirements.txt` ya incluye `pip-system-certs`, que hace que las verificaciones
  usen el almacén de certificados del sistema operativo. Asegurate de tenerlo instalado.

- **`umap-learn` o `bertviz` fallan al importar**: el notebook los maneja con try/except,
  así que el resto sigue funcionando. En la Sección 3 verás solo t-SNE y en la Sección 5
  se usará un heatmap con matplotlib en lugar de la vista interactiva.

- **El notebook no encuentra el kernel `.venv`**: instalá el kernel dentro del venv:
  ```powershell
  python -m ipykernel install --user --name curso-llm --display-name "Python (curso-llm)"
  ```

- **La activación del venv falla en PowerShell**: revisá el `Set-ExecutionPolicy` del
  Paso 2 de la sección del notebook.

- **La primera ejecución es lenta**: se descargan modelos (GPT-2, BERT, DistilBERT) y
  vectores GloVe (~130 MB). A partir de la segunda vez quedan cacheados.

- **`error: Unable to find a compatible Visual Studio installation` al instalar `greenlet`**
  (dependencia de LangChain/SQLAlchemy): pasa si el venv usa **Python 3.9** — versiones
  recientes de `greenlet` ya no publican wheel precompilado para 3.9 en Windows, y sin
  Visual Studio Build Tools no se puede compilar desde código fuente. Solución: recreá el
  venv con **Python 3.11** (`py -3.11 -m venv .venv`), que sí tiene wheels disponibles.

---

## 16. Notas

- Probado con **Python 3.11** en **Windows** (CPU, sin GPU).
- Las versiones de `requirements.txt` están fijadas para asegurar reproducibilidad.
- La carpeta `.venv/` y las cachés (`chroma_db/`, `__pycache__/`, etc.) están ignoradas
  por Git (ver `.gitignore`).
