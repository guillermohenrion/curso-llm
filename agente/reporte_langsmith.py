"""
Genera un REPORTE HTML analizando la ULTIMA corrida del agente en LangSmith.

Toma la ultima corrida raiz (root run) de un proyecto de LangSmith, reconstruye
el arbol de pasos (el ciclo ReAct: llamadas al LLM + ejecuciones de tools) y
arma un archivo HTML con:
    - Resumen (pregunta, respuesta final, estado, duracion, tokens)
    - Analisis ReAct (cuantas iteraciones, que tools se usaron y cuantas veces)
    - Linea de tiempo de los pasos (LLM y tools), con sus inputs/outputs

Es SOLO LECTURA: usa list_runs con las credenciales del .env, no modifica nada.

Requisitos:
    - Mismo .env que agente_completo.py (LANGSMITH_API_KEY, LANGSMITH_PROJECT).
    - pip install langsmith python-dotenv  (ya estan en requirements-agente.txt)

Uso:
    python reporte_langsmith.py
    python reporte_langsmith.py curso-llm-rag_agente_react       # proyecto puntual
    python reporte_langsmith.py curso-llm-rag_agente_react salida.html
"""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Proyecto por defecto: donde el agente registra sus corridas ReAct.
PROYECTO_DEFAULT = os.getenv("LANGSMITH_PROJECT") or "curso-llm-rag_agente_react"
SALIDA_DEFAULT = os.path.join(os.path.dirname(__file__), "reporte_langsmith.html")

# Diagrama del ciclo ReAct (SVG inline -> el HTML queda autocontenido, sin imagen externa).
DIAGRAMA_REACT = """
<svg viewBox="0 0 800 250" width="100%" style="max-width:760px;display:block;margin:8px auto"
     font-family="system-ui, Segoe UI, sans-serif" role="img"
     aria-label="Ciclo ReAct: Thought, Action, Observation, en bucle, y salida a Final Answer">
  <defs>
    <marker id="fl" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#6b7280"/>
    </marker>
  </defs>
  <!-- Thought -->
  <rect x="30" y="30" width="200" height="62" rx="10" fill="#ede9fe" stroke="#c4b5fd"/>
  <text x="130" y="56" text-anchor="middle" font-weight="700" fill="#5b21b6" font-size="15">1. Thought</text>
  <text x="130" y="76" text-anchor="middle" fill="#6b21a8" font-size="12">el LLM razona qué hacer</text>
  <!-- Action -->
  <rect x="300" y="30" width="200" height="62" rx="10" fill="#ede9fe" stroke="#c4b5fd"/>
  <text x="400" y="56" text-anchor="middle" font-weight="700" fill="#5b21b6" font-size="15">2. Action + Input</text>
  <text x="400" y="76" text-anchor="middle" fill="#6b21a8" font-size="12">elige una tool y su entrada</text>
  <!-- Observation -->
  <rect x="570" y="30" width="200" height="62" rx="10" fill="#dcfce7" stroke="#86efac"/>
  <text x="670" y="56" text-anchor="middle" font-weight="700" fill="#166534" font-size="15">3. Observation</text>
  <text x="670" y="76" text-anchor="middle" fill="#15803d" font-size="12">resultado de la tool</text>
  <!-- Final Answer -->
  <rect x="570" y="165" width="200" height="54" rx="10" fill="#dbeafe" stroke="#93c5fd"/>
  <text x="670" y="190" text-anchor="middle" font-weight="700" fill="#1e40af" font-size="15">Final Answer</text>
  <text x="670" y="208" text-anchor="middle" fill="#1d4ed8" font-size="12">respuesta al usuario</text>
  <!-- flechas del ciclo -->
  <line x1="230" y1="61" x2="292" y2="61" stroke="#6b7280" stroke-width="2" marker-end="url(#fl)"/>
  <line x1="500" y1="61" x2="562" y2="61" stroke="#6b7280" stroke-width="2" marker-end="url(#fl)"/>
  <text x="365" y="24" text-anchor="middle" fill="#9ca3af" font-size="11">el executor ejecuta la tool</text>
  <!-- loop-back: Observation -> Thought -->
  <path d="M600,92 L600,140 L130,140 L130,94" fill="none" stroke="#6b7280"
        stroke-width="2" marker-end="url(#fl)"/>
  <text x="360" y="134" text-anchor="middle" fill="#6b7280" font-size="11">
    si falta info: otro ciclo — la Observation vuelve al Thought</text>
  <!-- salida a Final Answer -->
  <line x1="670" y1="92" x2="670" y2="163" stroke="#2563eb" stroke-width="2" marker-end="url(#fl)"/>
  <text x="682" y="130" text-anchor="start" fill="#2563eb" font-size="11">si ya tiene todo</text>
</svg>
"""


# ---------------------------------------------------------------------------
# Extraccion de datos de las corridas
# ---------------------------------------------------------------------------
def dur(run) -> float:
    """Duracion de un run en segundos (0 si falta algun timestamp)."""
    if run.end_time and run.start_time:
        return (run.end_time - run.start_time).total_seconds()
    return 0.0


def texto_llm(run) -> str:
    """Best-effort: extrae el texto generado por una llamada al LLM."""
    out = run.outputs or {}
    try:
        gens = out.get("generations")
        if gens:
            g0 = gens[0][0] if isinstance(gens[0], list) else gens[0]
            if g0.get("text"):
                return g0["text"]
            msg = g0.get("message", {})
            contenido = msg.get("kwargs", {}).get("content") or msg.get("content")
            if contenido:
                return contenido
    except Exception:  # noqa: BLE001
        pass
    return str(out)


def texto_io(valor) -> str:
    """Normaliza inputs/outputs de tools (dict o string) a texto legible."""
    if isinstance(valor, dict):
        # Casos tipicos: {"input": "..."} / {"output": "..."} / {"query": "..."}
        for k in ("input", "output", "query", "expresion", "texto"):
            if k in valor:
                return str(valor[k])
        return str(valor)
    return str(valor)


def tokens_de(run) -> int:
    """Tokens totales de un run de LLM (0 si el proveedor no los reporta)."""
    return getattr(run, "total_tokens", None) or 0


# ---------------------------------------------------------------------------
# Armado del HTML
# ---------------------------------------------------------------------------
def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def explicacion_react(orden: list) -> str:
    """Arma un recorrido paso a paso del ciclo ReAct usando las tools reales de la corrida."""
    tools_ord = [r for r in orden if r.run_type == "tool"]
    items = []
    for i, r in enumerate(tools_ord, 1):
        entrada = texto_io(r.inputs)
        obs = texto_io(r.outputs) if r.outputs else "(sin salida)"
        obs = (obs[:200] + "…") if len(obs) > 200 else obs
        items.append(
            f"<li><b>Iteracion {i}</b> — <i>Thought</i>: el modelo decide que le falta algo "
            f"y elige una <i>Action</i>: <code>{_esc(r.name)}</code> con "
            f"<i>Action Input</i> <code>{_esc(entrada)}</code>. "
            f"El sistema ejecuta la tool y le devuelve la <i>Observation</i>: "
            f"<code>{_esc(obs)}</code>.</li>"
        )
    return "\n".join(items)


def _fila_paso(i: int, run) -> str:
    tipo = run.run_type
    if tipo == "llm":
        detalle = texto_llm(run)
    elif tipo == "tool":
        entrada = texto_io(run.inputs)
        salida = texto_io(run.outputs) if run.outputs else "(sin salida)"
        detalle = f"Input: {entrada}\nObservation: {salida}"
    else:
        detalle = texto_io(run.outputs) if run.outputs else ""
    clase = f"tipo-{tipo}"
    return (
        f"<tr class='{clase}'>"
        f"<td class='num'>{i}</td>"
        f"<td><span class='badge {clase}'>{_esc(tipo)}</span></td>"
        f"<td class='nombre'>{_esc(run.name)}</td>"
        f"<td class='dur'>{dur(run):.2f}s</td>"
        f"<td><pre>{_esc(detalle)}</pre></td>"
        f"</tr>"
    )


def construir_html(root, hijos: list, proyecto: str) -> str:
    pregunta = texto_io((root.inputs or {}).get("input", root.inputs))
    respuesta = texto_io((root.outputs or {}).get("output", root.outputs)) if root.outputs else "(sin salida / error)"

    orden = sorted(hijos, key=lambda r: r.start_time or datetime.min)
    # Pasos "de interes" para ReAct: llamadas al LLM (Thought/Action) y tools (Observation).
    pasos = [r for r in orden if r.run_type in ("llm", "tool")]

    llm_runs = [r for r in hijos if r.run_type == "llm"]
    tool_runs = [r for r in hijos if r.run_type == "tool"]
    tokens_total = sum(tokens_de(r) for r in llm_runs)

    # Conteo de tools usadas (nombre -> veces).
    conteo_tools: dict[str, int] = {}
    for r in tool_runs:
        conteo_tools[r.name] = conteo_tools.get(r.name, 0) + 1
    tools_txt = ", ".join(f"{n} (x{c})" for n, c in sorted(conteo_tools.items())) or "ninguna"

    url = f"https://smith.langchain.com/o/-/projects/p/{root.session_id}/r/{root.id}"

    narrativa = explicacion_react(orden)
    diagrama = DIAGRAMA_REACT
    filas = "\n".join(_fila_paso(i + 1, r) for i, r in enumerate(pasos))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte LangSmith — {_esc(root.name)}</title>
<style>
  body {{ font-family: system-ui, Segoe UI, Roboto, sans-serif; margin: 0; background: #f5f6f8; color: #1c2330; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 20px; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 18px; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .metric {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; text-align: center; }}
  .metric .n {{ font-size: 26px; font-weight: 700; }}
  .metric .l {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }}
  .qa p {{ margin: 6px 0; }}
  .qa .lbl {{ font-weight: 700; color: #374151; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; }}
  th, td {{ border-bottom: 1px solid #eef0f3; padding: 8px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
  th {{ background: #f0f2f5; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
  td.num {{ color: #9ca3af; width: 34px; }}
  td.dur {{ white-space: nowrap; color: #6b7280; width: 64px; }}
  td.nombre {{ font-weight: 600; white-space: nowrap; }}
  pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, Consolas, monospace; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
  .badge.tipo-llm {{ background: #ede9fe; color: #5b21b6; }}
  .badge.tipo-tool {{ background: #dcfce7; color: #166534; }}
  tr.tipo-tool {{ background: #fafffb; }}
  a {{ color: #2563eb; }}
  .estado-success {{ color: #166534; font-weight: 700; }}
  .estado-error {{ color: #b91c1c; font-weight: 700; }}
  .explica {{ background: #fbfaff; border-color: #ddd6fe; }}
  .explica h2 {{ font-size: 17px; margin: 0 0 10px; color: #5b21b6; }}
  .explica ul, .explica ol {{ margin: 8px 0; padding-left: 22px; }}
  .explica li {{ margin: 6px 0; }}
  .loop {{ background: #ede9fe; color: #5b21b6; border-radius: 8px; padding: 10px 12px;
           font-weight: 600; font-size: 13px; text-align: center; margin: 10px 0; }}
  code {{ background: #eef0f3; padding: 1px 5px; border-radius: 4px;
          font-family: ui-monospace, Consolas, monospace; font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Reporte de corrida — {_esc(root.name)}</h1>
  <div class="sub">Proyecto <b>{_esc(proyecto)}</b> · {_esc(root.start_time)} ·
    <a href="{_esc(url)}" target="_blank">ver en LangSmith ↗</a></div>

  <div class="grid">
    <div class="metric"><div class="n">{len(llm_runs)}</div><div class="l">Iteraciones (LLM)</div></div>
    <div class="metric"><div class="n">{len(tool_runs)}</div><div class="l">Tools ejecutadas</div></div>
    <div class="metric"><div class="n">{dur(root):.1f}s</div><div class="l">Duracion total</div></div>
    <div class="metric"><div class="n">{tokens_total or '—'}</div><div class="l">Tokens</div></div>
  </div>

  <div class="card qa">
    <p><span class="lbl">Estado:</span>
       <span class="estado-{_esc(root.status)}">{_esc(root.status)}</span></p>
    <p><span class="lbl">Pregunta:</span> {_esc(pregunta)}</p>
    <p><span class="lbl">Respuesta final:</span> {_esc(respuesta)}</p>
    <p><span class="lbl">Tools usadas:</span> {_esc(tools_txt)}</p>
  </div>

  <div class="card explica">
    <h2>El patron ReAct explicado con esta corrida</h2>
    <p><b>ReAct</b> (Reasoning + Acting) es la forma en que este agente resuelve
    tareas. Un LLM "pelado" contesta de una sola vez con lo que sabe; un agente
    ReAct, en cambio, entra en un <b>bucle</b>: <b>piensa</b> qué le falta,
    <b>actua</b> usando una herramienta, <b>observa</b> el resultado, y usa esa
    observacion para pensar el paso siguiente. Repite ese ciclo hasta reunir todo lo
    necesario y recien ahi da la respuesta final. Así el modelo puede consultar
    datos, hacer cuentas o ejecutar acciones en vez de depender solo de su memoria.</p>

    <div class="loop">Thought (pienso) → Action (uso una tool) → Observation (miro el resultado) → … → Final Answer</div>

    {diagrama}

    <p>Tres ideas clave para entenderlo:</p>
    <ul>
      <li>El <b>LLM no ejecuta</b> las herramientas: solo <i>escribe</i> qué accion
      quiere (<code>Action</code> + <code>Action Input</code>). Quien la ejecuta es
      el programa (el <i>executor</i>), que le devuelve la <code>Observation</code>.</li>
      <li>Cada observacion se <b>acumula</b> en el historial del razonamiento, así el
      modelo "recuerda" lo que ya hizo y no repite pasos.</li>
      <li>El bucle termina cuando el modelo escribe <code>Final Answer</code> (o al
      llegar al tope de iteraciones, para no quedar en loop).</li>
    </ul>

    <p>En <b>esta corrida</b>, la pregunta tenia varias partes, así que el agente
    hizo <b>{len(tool_runs)} acciones</b> (una por parte) a lo largo de
    <b>{len(llm_runs)} iteraciones</b> del LLM:</p>
    <ol>
      {narrativa}
    </ol>
    <p>Recien despues de cubrir todas las partes, en la ultima iteracion el modelo
    escribio la <b>Final Answer</b> combinando todo lo observado (ver el ultimo paso
    <span class="badge tipo-llm">llm</span> de la tabla de abajo).</p>

    <p class="sub" style="margin:0">En la linea de tiempo:
    <span class="badge tipo-llm">llm</span> = el modelo razonando y eligiendo la accion;
    <span class="badge tipo-tool">tool</span> = la herramienta ejecutandose y su Observation.</p>
  </div>

  <table>
    <thead><tr><th>#</th><th>Tipo</th><th>Nombre</th><th>Dur.</th><th>Detalle</th></tr></thead>
    <tbody>
      {filas}
    </tbody>
  </table>

  <p class="sub" style="margin-top:16px">Generado por reporte_langsmith.py · solo lectura de LangSmith.</p>
</div>
</body>
</html>"""


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("[error] No hay LANGSMITH_API_KEY en el .env. Copia .env.example a .env "
              "y completa tu API key.")
        sys.exit(1)

    proyecto = sys.argv[1] if len(sys.argv) > 1 else PROYECTO_DEFAULT
    salida = sys.argv[2] if len(sys.argv) > 2 else SALIDA_DEFAULT

    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError
    client = Client()

    print(f"Buscando la ultima corrida del proyecto '{proyecto}'...")
    try:
        roots = list(client.list_runs(project_name=proyecto, is_root=True, limit=1))
    except LangSmithNotFoundError:
        disponibles = [p.name for p in client.list_projects(limit=50)]
        print(f"[error] El proyecto '{proyecto}' no existe en tu LangSmith.")
        print(f"        Proyectos disponibles: {', '.join(disponibles) or '(ninguno)'}")
        print("        Pasá el nombre correcto: python reporte_langsmith.py <proyecto>")
        sys.exit(1)
    if not roots:
        print(f"[error] No hay corridas en el proyecto '{proyecto}'. "
              "Verifica el nombre (LANGSMITH_PROJECT) o corre el agente con tracing activo.")
        sys.exit(1)
    root = roots[0]

    # Traemos todo el arbol de la traza (los pasos hijos del root).
    try:
        hijos = list(client.list_runs(project_name=proyecto, trace_id=root.trace_id))
    except TypeError:
        # Compatibilidad con versiones viejas del SDK.
        hijos = list(client.list_runs(project_name=proyecto,
                                      filter=f'eq(trace_id, "{root.trace_id}")'))

    html_txt = construir_html(root, hijos, proyecto)
    with open(salida, "w", encoding="utf-8") as f:
        f.write(html_txt)

    print(f"[ok] Reporte generado: {salida}")
    print(f"     Corrida: {root.name} ({root.status}) — {len(hijos)} pasos en la traza.")


if __name__ == "__main__":
    main()
