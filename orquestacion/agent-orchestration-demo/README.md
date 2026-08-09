# Agent Orchestration Demo

Proyecto **educativo** en Python que muestra, de la forma más simple
posible, dos patrones fundamentales para orquestar agentes de IA:

1. **Pipeline secuencial** — varios agentes encadenados, cada uno
   procesando la salida del anterior.
2. **Router / Enrutador** — un agente que decide dinámicamente qué
   agente(s) especializado(s) deben resolver una solicitud.

No usa LangChain, LangGraph, CrewAI, AutoGen, MCP, bases vectoriales,
Docker, AWS ni API keys. Todo corre en local, con un `MockLLM` que
simula respuestas de forma determinística.

```
python 3.11+
sin frameworks de agentes
sin infraestructura externa
```

---

## 1. ¿Qué es un agente?

Un **agente** es, en su forma más simple, código que:

1. recibe un **input** (una solicitud, datos, contexto),
2. arma un **prompt** con instrucciones + esos datos,
3. se lo pasa a un **LLM** (`llm.generate(prompt)`),
4. interpreta la respuesta del LLM,
5. devuelve un **output** estructurado (texto, lista, dict...).

En este proyecto, un agente es literalmente una clase con un método
`run(...)` que hace esos cinco pasos. No hay magia: el "agente" es el
código que rodea al LLM, no el LLM en sí.

## 2. ¿Qué es un agente especializado?

Es un agente cuyo **prompt** y **responsabilidad** están acotados a una
tarea muy concreta. En vez de un agente genérico que "hace de todo",
tenemos varios agentes chicos y enfocados:

- `TitleAgent` solo genera títulos.
- `CreditAgent` solo responde sobre créditos y préstamos.

Esto hace que cada agente sea más simple, más fácil de testear y más
fácil de razonar sobre él — la misma idea que "funciones pequeñas con
una sola responsabilidad", aplicada a agentes.

## 3. ¿Qué es un orquestador?

Es el código que **decide en qué orden y con qué datos** se ejecutan los
agentes. El orquestador no resuelve la tarea él mismo: solo coordina.
En este proyecto hay dos orquestadores distintos, uno por patrón:

- `PipelineOrchestrator` — orden fijo, siempre A1 → A2 → A3.
- `RouterAgent` + `Aggregator` — orden dinámico, decidido en tiempo de
  ejecución según el contenido de la solicitud.

---

## 4. Cómo funciona el Pipeline

```
                 SOLICITUD DEL USUARIO
                          |
                          v
                +-------------------+
                |   A1 TitleAgent   |   Prompt P1: genera títulos
                +-------------------+
                          |
                     [ titles ]
                          v
                +-------------------+
                |  A2 ContentAgent  |   Prompt P2: genera contenido
                +-------------------+   (recibe: request + titles)
                          |
                    [ sections ]
                          v
                +-------------------+
                |  A3 EditorAgent   |   Prompt P3: pule títulos y
                +-------------------+   arma el documento final
                          |
                          v
                   RESULTADO FINAL
```

Cada agente **no sabe qué viene después**. Solo el orquestador
(`pipeline/orchestrator.py`) conoce la secuencia completa:

```python
result_1 = title_agent.run(user_request)
result_2 = content_agent.run(user_request, result_1)
result_3 = editor_agent.run(user_request, result_1, result_2)
return result_3
```

Ejecutar:

```bash
python pipeline/main.py
```

La consola muestra, para cada etapa: el **PROMPT** completo enviado al
LLM, el **INPUT** recibido y el **OUTPUT** producido — así se puede ver
exactamente cómo el resultado de A1 se convierte en el input de A2, y el
de A2 en el input de A3.

## 5. Cómo funciona el Router

```
                       +--> CreditAgent
                       |
USER --> RouterAgent --+--> InvestmentAgent
                       |
                       +--> CardAgent
                               |
                               v
                          Aggregator
                               |
                               v
                        FINAL RESPONSE
```

A diferencia del pipeline, aquí **no hay un camino fijo**. El
`RouterAgent` lee la solicitud, la compara contra las `capabilities` de
cada agente del catálogo (`AGENT_REGISTRY`) y decide cuáles ejecutar. Si
más de un agente aplica (ej. "comparar un préstamo con una inversión"),
el router los selecciona a todos y se ejecutan **en paralelo** con
`ThreadPoolExecutor`. El `Aggregator` combina las respuestas en una
única `FinalResponse`.

Ejecutar:

```bash
python router/main.py
```

Corre 4 casos: préstamo (1 agente), inversión (1 agente), préstamo +
inversión (2 agentes en paralelo) y tarjeta (1 agente).

## 6. Diferencia entre Pipeline y Router

| | Pipeline | Router |
|---|---|---|
| Camino | Fijo, definido en el código | Dinámico, decidido en tiempo de ejecución |
| Nº de agentes que corren | Siempre todos, en orden | Solo los que aplican, en paralelo si son varios |
| Quién decide | El orquestador (hardcoded) | El RouterAgent (según el input) |
| Analogía | Una línea de ensamblaje | Un conmutador/switchboard |

> **Pipeline:** *"La arquitectura define el camino."*
> **Router:** *"El Router decide el camino."*

```
PIPELINE                          ROUTER

  User                                          +--> A1
   |                                            |
   v                             User -> Router --> A2
   A1                                            |
   |                                            +--> A3
   v
   A2
   |
   v
   A3
   |
   v
 Result
```

## 7. Cómo se pasan los resultados

En el **pipeline**, el resultado de un agente se pasa **explícitamente**
como argumento al siguiente (`content_agent.run(request, titles)`). Es
un traspaso directo, síncrono y ordenado.

En el **router**, cada agente seleccionado recibe la **misma** solicitud
original (no el resultado de otro agente) y corre de forma
**independiente**. Sus resultados (`AgentResult`) se juntan recién al
final, en el `Aggregator`. No hay traspaso entre agentes especializados.

## 8. Cómo funciona el Agent Registry

`AGENT_REGISTRY` (en `router/router.py`) es una lista de diccionarios
—el catálogo de agentes disponibles— con `name`, `description` y
`capabilities` (palabras clave). El `RouterAgent` **solo** lee este
catálogo para decidir a quién enrutar; no conoce ni importa las clases
de agentes reales (`CreditAgent`, etc.). Esa separación permite, por
ejemplo, agregar un agente nuevo al catálogo sin tocar la lógica de
enrutamiento.

## 9. Cómo decide el Router

Estrategia v1: **keyword matching**. El texto de la solicitud se
normaliza (minúsculas, sin acentos) y se busca si contiene alguna de las
`capabilities` de cada agente. Si varias coinciden, se seleccionan
varios agentes. Es intencionalmente simple — ver punto 13 para cómo
evolucionarla.

## 10. Cómo se ejecutan agentes en paralelo

Cuando el `RoutingDecision` selecciona más de un agente,
`router/main.py` usa:

```python
with ThreadPoolExecutor(max_workers=len(selected)) as executor:
    futures = [executor.submit(agent.run, user_request) for agent in selected]
    results = [future.result() for future in futures]
```

Todos los `run()` se **envían** al pool antes de esperar ningún
resultado, por lo que corren concurrentemente. Con un LLM real (con
latencia de red) esto reduce el tiempo total de espera de "suma de
tiempos" a "el más lento de todos".

## 11. Cómo funciona el Aggregator

Recibe la lista de `AgentResult` (uno por agente ejecutado) y produce
una `FinalResponse`. En esta v1 **no usa un LLM**: si hay un solo
resultado lo devuelve tal cual; si hay varios, los concatena indicando
de qué agente viene cada parte. El código deja marcado explícitamente
dónde iría la mejora:

```python
# FUTURE:
# Replace this aggregation logic with an LLM
# that synthesizes the individual agent responses.
```

## 12. Cómo reemplazar MockLLM por Claude (u otro LLM)

Todo el proyecto habla con el LLM a través de una única interfaz:

```python
class MockLLM:
    def generate(self, prompt: str) -> str:
        ...
```

Para usar un LLM real, basta con crear una clase con la misma interfaz:

```python
class ClaudeLLM:
    def __init__(self, client, model="claude-sonnet-5"):
        self.client = client
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

Y pasarla al construir los agentes (`TitleAgent(ClaudeLLM(...))`,
`CreditAgent(ClaudeLLM(...))`, etc.) en vez de `MockLLM()`. Ningún otro
archivo del proyecto necesita cambiar: agentes, orquestador, router y
aggregator solo dependen de `generate(prompt) -> str`.

> Nota: con un LLM real, los agentes deberían pedir explícitamente su
> salida en un formato parseable (por ejemplo JSON) y parsearla, en vez
> de depender de los marcadores de texto fijos que usa `MockLLM` para
> simular determinismo.

## 13. Cómo evolucionar el Router hacia LLM + embeddings + RAG

El router actual (keyword matching) es un punto de partida. Camino de
evolución típico:

1. **Router basado en LLM**: en vez de buscar palabras clave, se le pasa
   al LLM la solicitud + el catálogo de agentes (nombre + descripción) y
   se le pide que elija cuáles aplican. Más flexible ante frases que no
   usan las palabras clave exactas ("necesito plata para comprar un
   auto" → debería enrutar a `credit_agent` aunque no diga "préstamo").
2. **Embeddings**: se calcula el embedding de la solicitud y el de la
   `description`/`capabilities` de cada agente, y se enruta por
   similitud coseno. Permite catálogos grandes sin que el prompt del
   router crezca sin límite.
3. **RAG sobre el catálogo**: si el catálogo de agentes es muy extenso
   (decenas o cientos), se indexa en una base vectorial y se recuperan
   solo los agentes candidatos más relevantes antes de decidir, en vez
   de comparar contra todo el catálogo en cada solicitud.

Estos tres pasos son incrementales: cada uno reemplaza `RouterAgent.route()`
manteniendo la misma interfaz (`user_request -> RoutingDecision`), por lo
que el resto del sistema (agentes, aggregator, main) no cambia.

---

## Estructura del proyecto

```
agent-orchestration-demo/
├── README.md
│
├── common/
│   ├── __init__.py
│   ├── llm_mock.py        # MockLLM: simula el LLM de forma determinística
│   └── models.py          # AgentRequest, AgentResult, RoutingDecision, FinalResponse
│
├── pipeline/
│   ├── __init__.py
│   ├── agents.py           # TitleAgent (A1), ContentAgent (A2), EditorAgent (A3)
│   ├── orchestrator.py     # PipelineOrchestrator: A1 -> A2 -> A3
│   └── main.py              # python pipeline/main.py
│
└── router/
    ├── __init__.py
    ├── agents.py            # CreditAgent, InvestmentAgent, CardAgent
    ├── router.py            # AgentRegistry + RouterAgent (keyword routing)
    ├── aggregator.py        # Aggregator: combina resultados de varios agentes
    └── main.py               # python router/main.py
```

## Cómo ejecutar

Desde la carpeta `agent-orchestration-demo/`:

```bash
python pipeline/main.py
python router/main.py
```

No requieren dependencias externas ni variables de entorno — solo
Python 3.11+ y la librería estándar (`dataclasses`, `concurrent.futures`,
`time`, `re`).
