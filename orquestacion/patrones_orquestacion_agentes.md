# Patrones de Orquestación de Agentes

## 1. Introducción

Cuando hablamos de **orquestación de agentes** estamos describiendo la
forma en que varios agentes especializados colaboran para resolver una
solicitud compleja.

Un agente no necesariamente tiene que resolver todo el problema. Una
arquitectura más robusta puede dividir el trabajo en agentes
especializados y definir cómo:

-   se inicia el flujo;
-   se decide qué agente participa;
-   qué información recibe cada agente;
-   cómo se ejecuta cada tarea;
-   cómo se transmiten los resultados;
-   cómo se combinan los resultados;
-   y cómo se devuelve la respuesta final al usuario.

En este documento se describen dos patrones:

1.  **Orquestación secuencial / Pipeline de agentes**.
2.  **Orquestación mediante Agente Enrutador / Router Agent**.

------------------------------------------------------------------------

# 2. Patrón 1 --- Orquestación secuencial de agentes

## 2.1. Idea principal

En este patrón el problema se divide en una **cadena de tareas
especializadas**.

Cada agente tiene una responsabilidad concreta y recibe como entrada:

-   la solicitud original;
-   el resultado producido por agentes anteriores;
-   un prompt específico para su tarea;
-   y eventualmente contexto adicional.

El flujo puede representarse como:

``` text
Usuario
   |
   v
  A1
  P1
   |
   | Resultado 1
   v
  A2
  P2
   |
   | Resultado 2
   v
  A3
  P3
   |
   v
Resultado final
```

Una característica importante es que **el orden de ejecución está
definido por la arquitectura**.

No es A1 quien decide si debe ejecutarse A2. El orquestador conoce
previamente el flujo:

``` text
A1 -> A2 -> A3
```

------------------------------------------------------------------------

## 2.2. Ejemplo

Supongamos que queremos generar un documento.

La solicitud del usuario es:

> "Preparar un documento sobre inteligencia artificial aplicada al
> sector financiero."

Podemos dividir el problema en tres agentes.

### Agente A1 --- Generador de títulos

Responsabilidad:

> Generar la estructura temática y los títulos principales.

Prompt:

``` text
P1:
A partir de la solicitud del usuario, genera una lista
de títulos y subtítulos para estructurar el documento.
No desarrolles todavía el contenido.
```

Salida:

``` text
1. Introducción a la IA en finanzas
2. Casos de uso
3. Riesgos
4. Gobernanza
5. Conclusiones
```

------------------------------------------------------------------------

### Agente A2 --- Generador de contenido

A2 recibe los títulos generados por A1.

``` text
A1
 |
 | títulos
 v
A2
```

Prompt:

``` text
P2:
Utilizando los títulos proporcionados, desarrolla
el contenido correspondiente a cada sección.
Mantén coherencia entre las secciones.
```

Salida:

``` text
Título 1:
Introducción a la IA en finanzas

Contenido:
...

Título 2:
Casos de uso

Contenido:
...
```

------------------------------------------------------------------------

### Agente A3 --- Editor / refinador

A3 recibe el documento generado y puede encargarse de mejorar la
estructura final.

``` text
A2
 |
 | contenido
 v
A3
 |
 | documento final
 v
Usuario
```

Prompt:

``` text
P3:
Revisa el documento generado.
Mejora los títulos para que sean claros,
consistentes y adecuados al contenido.
No modifiques innecesariamente el contenido.
```

------------------------------------------------------------------------

# 2.3. Característica fundamental: el contexto se propaga

El resultado de un agente se convierte en entrada del siguiente.

Formalmente:

``` text
R1 = A1(P1, solicitud)

R2 = A2(P2, solicitud, R1)

R3 = A3(P3, solicitud, R1, R2)
```

Por lo tanto:

``` text
A1 -> R1 -> A2 -> R2 -> A3 -> R3
```

El orquestador puede decidir exactamente qué información de cada etapa
pasa a la siguiente.

Esto es importante porque **no necesariamente hay que enviar todo el
historial de la conversación a cada agente**.

Por ejemplo:

``` text
A1 -> genera títulos

A2 recibe:
- solicitud original
- títulos

A3 recibe:
- solicitud original
- títulos
- contenido
```

También se puede resumir o transformar el resultado antes de pasarlo al
siguiente agente.

------------------------------------------------------------------------

# 2.4. Ventajas

### Especialización

Cada agente tiene un objetivo pequeño y claro.

Esto permite utilizar:

-   diferentes prompts;
-   diferentes herramientas;
-   diferentes modelos;
-   diferentes fuentes de conocimiento;
-   diferentes reglas de validación.

### Predictibilidad

El flujo es conocido:

``` text
A1 -> A2 -> A3
```

Esto facilita:

-   debugging;
-   observabilidad;
-   testing;
-   auditoría;
-   medición de costos.

### Control

El sistema puede imponer reglas entre etapas.

Por ejemplo:

``` text
A1
 |
 v
Validación
 |
 +-- error --> A1 nuevamente
 |
 +-- OK -----> A2
```

### Composición

Un agente puede ser reemplazado sin modificar toda la arquitectura.

Por ejemplo:

``` text
A1 -> A2 -> A3
```

puede convertirse en:

``` text
A1 -> A2 -> A3 -> A4
```

------------------------------------------------------------------------

# 2.5. Desventajas

El principal problema es que el flujo está **predefinido**.

Si llega una solicitud que no necesita A2 o A3, igualmente podrían
ejecutarse.

Además:

-   aumenta la latencia si las etapas son secuenciales;
-   aumenta el costo porque se realizan varias invocaciones;
-   un error temprano puede propagarse;
-   el diseño debe conocer previamente el proceso.

------------------------------------------------------------------------

# 2.6. Cuándo utilizarlo

Es especialmente apropiado cuando existe un proceso conocido.

Ejemplos:

``` text
Analizar -> Clasificar -> Generar -> Validar
```

``` text
Extraer -> Transformar -> Resumir -> Revisar
```

``` text
Planificar -> Ejecutar -> Validar -> Publicar
```

``` text
Generar títulos -> Generar contenido -> Editar documento
```

La regla práctica es:

> **Si el proceso se conoce de antemano y las etapas tienen dependencias
> claras, un pipeline de agentes suele ser una buena opción.**

------------------------------------------------------------------------

# 3. Patrón 2 --- Agente Enrutador (Router Agent)

## 3.1. Idea principal

En este patrón existe un agente central cuyo objetivo no es resolver
directamente el problema, sino **determinar qué agente especializado
debe resolverlo**.

La arquitectura es:

``` text
                 +--> A1 -> R1
                 |
Usuario -> Router+--> A2 -> R2
                 |
                 +--> A3 -> R3
                       |
                       v
                    Resultado
```

El Router funciona como una capa de decisión.

------------------------------------------------------------------------

# 3.2. Ejemplo

Supongamos un sistema bancario con agentes especializados:

``` text
A1 = Agente de créditos
A2 = Agente de inversiones
A3 = Agente de tarjetas
A4 = Agente de reclamos
```

El usuario pregunta:

> "¿Qué requisitos necesito para solicitar un préstamo?"

El Router analiza la solicitud y determina:

``` text
Intent:
    crédito

Agente seleccionado:
    A1
```

Entonces:

``` text
Usuario
   |
   v
Router
   |
   v
A1 Crédito
   |
   v
Respuesta
```

------------------------------------------------------------------------

# 3.3. ¿Cómo sabe el Router a quién enviar?

Esta es una de las partes más importantes de este patrón.

El Router necesita información sobre los agentes disponibles.

Una opción sencilla es un **catálogo de agentes**:

``` json
[
  {
    "name": "credit_agent",
    "description": "Resuelve consultas sobre préstamos y créditos",
    "capabilities": [
      "prestamos",
      "tasas",
      "requisitos",
      "simulaciones"
    ]
  },
  {
    "name": "card_agent",
    "description": "Resuelve consultas sobre tarjetas",
    "capabilities": [
      "tarjetas",
      "limites",
      "consumos",
      "bloqueos"
    ]
  }
]
```

El Router utiliza esta información junto con el prompt del usuario.

------------------------------------------------------------------------

# 3.4. Estrategias para seleccionar agentes

## A. Reglas

La opción más simple.

``` text
si contiene "préstamo"
    -> credit_agent

si contiene "tarjeta"
    -> card_agent
```

Ventaja:

-   simple;
-   rápido;
-   determinista.

Desventaja:

-   poco flexible;
-   difícil de mantener cuando crece el número de agentes.

------------------------------------------------------------------------

## B. Clasificación mediante LLM

El Router puede utilizar un LLM para clasificar la intención.

Ejemplo:

``` text
Solicitud:
"Quiero saber cuánto puedo pedir prestado."

Router:

{
  "agent": "credit_agent",
  "confidence": 0.94
}
```

El LLM funciona como un clasificador semántico.

------------------------------------------------------------------------

## C. Embeddings / recuperación semántica

El catálogo de agentes puede almacenarse como documentos vectorizados.

Por ejemplo:

``` text
Agente:
Credit Agent

Descripción:
Especialista en préstamos, créditos,
tasas y requisitos de financiación.
```

La solicitud se transforma en embedding y se busca qué descripción de
agente es semánticamente más cercana.

Esto puede verse como un:

> **RAG de agentes**

El Router recupera los agentes potencialmente adecuados y luego puede
utilizar un LLM para seleccionar entre ellos.

------------------------------------------------------------------------

## D. Reglas + LLM + contexto

En sistemas reales es frecuente combinar estrategias.

Por ejemplo:

``` text
Solicitud
   |
   v
Reglas de seguridad
   |
   v
Recuperación de agentes
   |
   v
LLM Router
   |
   v
Agente seleccionado
```

También pueden intervenir:

-   permisos;
-   costo;
-   disponibilidad;
-   prioridad;
-   contexto de conversación;
-   historial;
-   SLA;
-   dominio;
-   nivel de confianza.

------------------------------------------------------------------------

# 3.5. ¿Puede seleccionar más de un agente?

Sí.

El Router puede decidir:

``` text
Usuario:
"Quiero comparar un préstamo con una inversión."

Router:

A1 = Crédito
A2 = Inversiones
```

Entonces puede ejecutar:

``` text
        +--> A1 -> R1
Router -|
        +--> A2 -> R2
```

Los agentes pueden ejecutarse:

### En paralelo

``` text
        +--> A1
Router -|
        +--> A2
```

Luego:

``` text
R1 + R2
  |
  v
Síntesis
```

### En secuencia

``` text
Router -> A1 -> A2 -> Resultado
```

La elección depende de las dependencias entre tareas.

------------------------------------------------------------------------

# 3.6. ¿Cómo resuelven los agentes invocados?

El Router **no necesariamente hace el trabajo de los agentes**.

Cada agente especializado puede tener su propio:

``` text
System Prompt
      +
LLM
      +
Tools
      +
Knowledge
      +
Memory
      +
Business Rules
```

Por ejemplo:

``` text
Credit Agent

LLM
 |
 +-- Tool: consultar tasas
 +-- Tool: consultar productos
 +-- KB: documentación de créditos
 +-- Rules: políticas crediticias
```

El agente ejecuta su propio loop:

``` text
recibir tarea
     |
     v
razonar
     |
     v
usar herramientas
     |
     v
validar
     |
     v
generar resultado
```

Finalmente devuelve al Router un resultado estructurado.

Por ejemplo:

``` json
{
  "agent": "credit_agent",
  "status": "success",
  "answer": "El cliente puede solicitar...",
  "confidence": 0.91,
  "sources": ["credit_policy_2026"],
  "metadata": {
    "execution_time_ms": 850
  }
}
```

------------------------------------------------------------------------

# 3.7. ¿Cómo devuelve el Router la respuesta final?

El Router recibe:

``` text
R1
R2
R3
...
RN
```

Luego puede realizar una etapa de:

``` text
validación
     |
     v
filtrado
     |
     v
resolución de conflictos
     |
     v
síntesis
     |
     v
respuesta final
```

Por ejemplo:

``` text
A1 -> "La tasa es 42%"
A2 -> "La tasa es 44%"
```

El Router podría:

1.  detectar la inconsistencia;
2.  consultar fuentes;
3.  priorizar la fuente más confiable;
4.  volver a invocar un agente;
5.  o informar la discrepancia.

Por lo tanto, el Router puede actuar como **orquestador y agregador**,
aunque en arquitecturas más grandes estas responsabilidades pueden
separarse:

``` text
Router
  |
  v
Agentes
  |
  v
Aggregator / Synthesizer
  |
  v
Respuesta final
```

------------------------------------------------------------------------

# 4. Diferencia fundamental entre ambos patrones

  -----------------------------------------------------------------------
  Característica          Pipeline secuencial     Router Agent
  ----------------------- ----------------------- -----------------------
  Flujo                   Predefinido             Dinámico

  Decisión                Arquitectura            Router

  Agentes                 Etapas conocidas        Seleccionados según
                                                  solicitud

  Orden                   Normalmente fijo        Puede variar

  Paralelismo             Posible, pero menos     Muy natural
                          natural                 

  Especialización         Alta                    Alta

  Flexibilidad            Media                   Alta

  Predictibilidad         Alta                    Media

  Complejidad             Menor                   Mayor

  Latencia                Puede crecer por        Puede optimizarse con
                          secuencia               paralelo

  Caso ideal              Workflow conocido       Problemas
                                                  abiertos/dinámicos
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 5. Diferencia conceptual

Una forma sencilla de explicarlo es:

### Pipeline

> **"Yo sé cómo resolver el problema, pero cada etapa necesita un
> especialista diferente."**

``` text
A1 -> A2 -> A3 -> Resultado
```

### Router

> **"No sé de antemano qué especialista necesito; primero debo decidir
> quién puede resolver la solicitud."**

``` text
             -> A1
             -> A2
Usuario -> Router
             -> A3
             -> A4
```

------------------------------------------------------------------------

# 6. Los dos patrones pueden combinarse

En arquitecturas reales no es necesario elegir exclusivamente uno.

Un Router puede seleccionar un agente que internamente ejecute un
pipeline.

Ejemplo:

``` text
                    +--> Credit Agent
                    |       |
Usuario -> Router --+       +--> Analizar
                    |       +--> Simular
                    |       +--> Validar
                    |
                    +--> Investment Agent
```

Aquí tenemos:

``` text
Nivel 1:
Router dinámico

Nivel 2:
Pipeline especializado
```

Esta combinación permite construir sistemas multiagente más complejos
sin convertir el Router en un componente monolítico.

------------------------------------------------------------------------

# 7. Observabilidad recomendada

En ambos patrones conviene registrar:

``` text
request_id
conversation_id
agent_id
prompt/version
model
input
output
tools utilizadas
latencia
tokens
costo
errores
confidence
resultado
```

Para un pipeline:

``` text
request
  |
  +-- A1
  |    +-- input/output
  |
  +-- A2
  |    +-- input/output
  |
  +-- A3
       +-- input/output
```

Para un Router:

``` text
request
 |
 +-- Router
      |
      +-- decisión
      |
      +-- A1
      |
      +-- A3
      |
      +-- síntesis
```

Esto permite responder preguntas como:

-   ¿Por qué se eligió ese agente?
-   ¿Qué información recibió?
-   ¿Qué agente produjo el error?
-   ¿Cuánto costó cada etapa?
-   ¿Cuánto tiempo tomó?
-   ¿Cuántas veces se invocó un agente?
-   ¿El Router eligió correctamente?

------------------------------------------------------------------------

# 8. Prompt para Claude Code --- construir ejemplos

El siguiente prompt está pensado para pegar directamente en Claude Code.

``` text
Quiero que construyas un proyecto educativo en Python que implemente
DOS patrones de orquestación de agentes:

1. Pipeline secuencial de agentes.
2. Agente Router / Enrutador.

OBJETIVO

El proyecto debe ser SIMPLE, DIDÁCTICO y EJECUTABLE LOCALMENTE.

No quiero una implementación compleja ni dependiente de AWS.

Utiliza Python y una arquitectura limpia que permita entender claramente
cómo funciona la orquestación.

IMPORTANTE:
- El código debe funcionar sin API keys.
- Implementa un LLM mock/simple para simular las respuestas.
- No utilices LangChain ni otros frameworks de agentes en la primera versión.
- Quiero entender primero la arquitectura.
- Después deja preparada la estructura para reemplazar el mock por un LLM real.

ESTRUCTURA DEL PROYECTO

Crear:

agent-orchestration-demo/
│
├── README.md
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── agents.py
│   ├── orchestrator.py
│   └── main.py
│
├── router/
│   ├── __init__.py
│   ├── agents.py
│   ├── router.py
│   ├── aggregator.py
│   └── main.py
│
└── common/
    ├── __init__.py
    ├── models.py
    └── llm_mock.py


==================================================
PARTE 1 — PIPELINE SECUENCIAL
==================================================

Implementar este flujo:

Usuario
  |
  v
A1 - Generador de títulos
  |
  | títulos
  v
A2 - Generador de contenido
  |
  | contenido
  v
A3 - Editor de títulos
  |
  v
Documento final


A1:

Nombre:
TitleAgent

Prompt:
"Genera una lista de títulos para un documento sobre el tema recibido."

Input:
user_request

Output:

{
    "titles": [...]
}


A2:

Nombre:
ContentAgent

Recibe:
- user_request
- titles

Genera contenido para cada título.

Output:

{
    "sections": [
        {
            "title": "...",
            "content": "..."
        }
    ]
}


A3:

Nombre:
TitleEditorAgent

Recibe:
- títulos
- contenido

Mejora los títulos para que sean coherentes con el contenido.

Output:

{
    "final_sections": [...]
}


El orchestrator debe ejecutar:

A1 -> A2 -> A3

Mostrar claramente en consola:

[ORCHESTRATOR]
Starting pipeline

[A1]
Input:
Output:

[A2]
Input:
Output:

[A3]
Input:
Output:

[FINAL RESULT]
...


==================================================
PARTE 2 — ROUTER AGENT
==================================================

Implementar:

Usuario
   |
   v
Router
   |
   +----> CreditAgent
   |
   +----> InvestmentAgent
   |
   +----> CardAgent
   |
   v
Aggregator
   |
   v
Respuesta final


Crear tres agentes especializados:

CreditAgent:
- préstamos
- créditos
- tasas
- financiación

InvestmentAgent:
- inversiones
- bonos
- acciones
- fondos

CardAgent:
- tarjetas
- límites
- consumos
- bloqueos


El Router debe tener un catálogo de agentes.

Ejemplo:

agents = [
    {
        "name": "credit_agent",
        "description": "Especialista en préstamos y créditos",
        "capabilities": [...]
    },
    ...
]


Implementar inicialmente una selección sencilla basada en keywords.

Ejemplo:

"Quiero pedir un préstamo"
    -> CreditAgent

"Quiero invertir dinero"
    -> InvestmentAgent

"Quiero aumentar el límite de mi tarjeta"
    -> CardAgent


Pero diseñar la clase Router de forma que posteriormente pueda reemplazarse
la selección por:

- LLM
- embeddings
- RAG de agentes
- reglas de negocio


El Router debe devolver algo similar a:

{
    "selected_agents": [
        "credit_agent"
    ],
    "reason": "La consulta trata sobre préstamos"
}


IMPORTANTE:

Demostrar también una consulta que requiera DOS agentes.

Ejemplo:

"Quiero comparar un préstamo con una inversión."

El Router debe seleccionar:

[
    "credit_agent",
    "investment_agent"
]


Los agentes deben poder ejecutarse en paralelo utilizando
concurrent.futures.ThreadPoolExecutor.

Cada agente debe devolver:

{
    "agent": "...",
    "status": "success",
    "answer": "...",
    "metadata": {
        "execution_time_ms": ...
    }
}


==================================================
AGGREGATOR
==================================================

Crear una clase Aggregator.

Debe recibir:

[
    result_agent_1,
    result_agent_2,
    ...
]


Y producir una respuesta final.

En esta primera versión el Aggregator puede simplemente combinar
las respuestas.

Dejar claramente indicado dónde posteriormente podría utilizarse
un LLM para hacer la síntesis.

==================================================
MODELO DE DATOS
==================================================

Crear dataclasses para:

AgentRequest
AgentResult
RoutingDecision
FinalResponse


==================================================
MOCK LLM
==================================================

Crear:

common/llm_mock.py

con una clase:

MockLLM

que tenga:

generate(prompt)

y devuelva respuestas determinísticas.

El objetivo es poder ejecutar todo sin internet ni API keys.

==================================================
README
==================================================

Crear un README.md que explique:

1. Qué es un agente.
2. Qué es un orquestador.
3. Patrón Pipeline.
4. Patrón Router.
5. Diferencias entre ambos.
6. Cómo ejecutar cada ejemplo.
7. Ejemplos de entrada y salida.
8. Cómo reemplazar MockLLM por Claude.
9. Cómo reemplazar el Router por un LLM.
10. Cómo incorporar embeddings/RAG para descubrir agentes.
11. Cómo incorporar observabilidad.
12. Cómo agregar validaciones y retries.

Incluir diagramas ASCII.

==================================================
REQUISITOS DE CALIDAD
==================================================

- Código Python 3.11+.
- Type hints.
- Dataclasses.
- Funciones pequeñas.
- Comentarios solamente donde aporten valor.
- Sin dependencias innecesarias.
- Manejo básico de errores.
- Logging claro.
- Separación entre agentes, router y orchestrator.
- Código fácil de leer para alguien que está aprendiendo
  arquitectura de agentes.

AL FINAL

Ejecuta los dos ejemplos y verifica que funcionan.

Después explícame:

1. Qué archivos creaste.
2. Cómo funciona el Pipeline.
3. Cómo funciona el Router.
4. Qué parte debería reemplazarse por un LLM real.
5. Cómo evolucionar este ejemplo hacia una arquitectura
   multiagente productiva.
```

------------------------------------------------------------------------

# 9. Evolución del ejemplo hacia producción

Una vez entendido el ejemplo básico, se puede evolucionar
progresivamente:

``` text
Nivel 1
Mock LLM
   |
   v
Pipeline / Router
```

↓

``` text
Nivel 2
LLM real
   |
   v
Agentes especializados
```

↓

``` text
Nivel 3
Tools + APIs + Knowledge Base
```

↓

``` text
Nivel 4
Router + Agent Registry + RAG
```

↓

``` text
Nivel 5
Memory + Feedback + Observability
```

↓

``` text
Nivel 6
Guardrails + Evaluation + Human-in-the-loop
```

La idea importante es **no empezar directamente con un framework
complejo**. Primero conviene entender la mecánica fundamental:

``` text
PROMPT
   |
   v
ORCHESTRATOR
   |
   +--> AGENT
   |      |
   |      +--> LLM
   |      +--> TOOLS
   |      +--> KNOWLEDGE
   |
   v
RESULT
```

y luego agregar las capacidades necesarias.

------------------------------------------------------------------------

# 10. Resumen conceptual

Los dos patrones representan dos formas diferentes de coordinar
especialistas.

### Pipeline

``` text
Solicitud
   |
   v
A1
   |
   v
A2
   |
   v
A3
   |
   v
Resultado
```

**La arquitectura define el camino.**

### Router

``` text
             +--> A1
             |
Solicitud -> Router --> A2
             |
             +--> A3
```

**El Router decide el camino.**

Y ambos pueden combinarse:

``` text
                    +--> Pipeline A
                    |
Usuario -> Router --+--> Pipeline B
                    |
                    +--> Agente C
```

Ese último modelo es especialmente útil para arquitecturas empresariales
de agentes, donde un Router de alto nivel selecciona capacidades y cada
capacidad puede tener internamente su propio workflow, herramientas,
memoria, validaciones y loops de ejecución.
