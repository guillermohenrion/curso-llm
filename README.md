# Curso LLM - ITBA

Material práctico del curso de **Large Language Models (LLMs)**. Este repositorio
contiene dos bloques independientes:

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

## 6. Estructura del repositorio

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
│   └── langchain/                     #   RAG con LangChain (versiones incrementales)
│       ├── comun.py                   #     modulo compartido (modelos, corpus, retriever)
│       ├── 1_rag_basico.py            #     RAG minimo (LCEL)
│       ├── 2_rag_memoria.py           #     RAG conversacional con memoria avanzada
│       ├── 3_rag_tools.py             #     agente ReAct con tools
│       ├── 4_rag_skills.py            #     skills + router
│       ├── 5_rag_langsmith.py         #     RAG con tracing en LangSmith
│       └── requirements-langchain.txt #     dependencias de LangChain
├── requirements.txt                   # Dependencias del notebook
├── requirements-hf-remoto.txt         # Dependencias de la demo remota de HF
├── .env.example                       # Plantilla para el token de HF (copiar a .env)
└── README.md                          # Este archivo
```

---

## 7. Problemas frecuentes

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

---

## 8. Notas

- Probado con **Python 3.11** en **Windows** (CPU, sin GPU).
- Las versiones de `requirements.txt` están fijadas para asegurar reproducibilidad.
- La carpeta `.venv/` y las cachés (`chroma_db/`, `__pycache__/`, etc.) están ignoradas
  por Git (ver `.gitignore`).
