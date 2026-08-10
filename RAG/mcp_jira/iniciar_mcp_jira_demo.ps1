<#
    RAG + MCP de Jira: RAG local + tools de Jira en vivo (mcp-atlassian) para clase.
    Uso:
      .\iniciar_mcp_jira_demo.ps1                              -> modo interactivo
      .\iniciar_mcp_jira_demo.ps1 "issues abiertos de ABC"    -> una sola pregunta y listo
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Pregunta
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Paso($msg) {
    Write-Host ""
    Write-Host ">> $msg" -ForegroundColor Cyan
}

# 1. Verificar que Ollama esté corriendo, y si no, levantarlo
Write-Paso "Verificando servidor de Ollama..."
$ollamaOk = $false
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434" -UseBasicParsing -TimeoutSec 2 | Out-Null
    $ollamaOk = $true
} catch {
    $ollamaOk = $false
}

if (-not $ollamaOk) {
    Write-Host "Ollama no esta corriendo. Iniciandolo..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    $intentos = 0
    while (-not $ollamaOk -and $intentos -lt 15) {
        Start-Sleep -Seconds 1
        $intentos++
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:11434" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $ollamaOk = $true
        } catch {
            $ollamaOk = $false
        }
    }
    if (-not $ollamaOk) {
        Write-Host "No se pudo iniciar Ollama. Instalalo desde https://ollama.com o inicia 'ollama serve' manualmente." -ForegroundColor Red
        exit 1
    }
}
Write-Host "Ollama OK." -ForegroundColor Green

# 2. Verificar que los modelos necesarios estén descargados (llama3.1 hace tool-calling; gemma3 no)
Write-Paso "Verificando modelos (llama3.1, nomic-embed-text)..."
$modelosInstalados = (ollama list) -join "`n"

foreach ($modelo in @("llama3.1", "nomic-embed-text")) {
    if ($modelosInstalados -notmatch [regex]::Escape($modelo)) {
        Write-Host "Descargando $modelo (puede tardar unos minutos)..." -ForegroundColor Yellow
        ollama pull $modelo
    } else {
        Write-Host "  $modelo ya esta descargado." -ForegroundColor Green
    }
}

# 3. Verificar que 'uv' (uvx) esté instalado: lanza el servidor MCP de Atlassian
Write-Paso "Verificando 'uv' (necesario para uvx)..."
if (-not (Get-Command uvx -ErrorAction SilentlyContinue)) {
    Write-Host "No se encontro 'uvx'. Instalalo desde https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}
Write-Host "uv OK." -ForegroundColor Green

# 4. Verificar credenciales de Jira (si faltan, el agente arranca solo con RAG)
if (-not (Test-Path (Join-Path $ScriptDir ".env"))) {
    Write-Host "Aviso: no hay .env (copy .env.example .env y completa JIRA_URL/JIRA_USERNAME/JIRA_API_TOKEN). Arranca solo con RAG." -ForegroundColor Yellow
}

# 5. Activar el entorno virtual PROPIO de esta demo (no el de la raiz del repo).
#    Motivo: langchain 0.3.x (create_tool_calling_agent) exige langchain-core<1.0,
#    pero las demos de LangGraph en el venv raiz ya subieron langchain-core a 1.x
#    (langgraph>=1.2 exige langchain-core>=1.4.7). Ambos requisitos son incompatibles
#    en un mismo entorno, asi que esta demo vive en su propio venv local.
Write-Paso "Activando entorno virtual local (RAG/mcp_jira/.venv)..."
$activate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "No se encontro el venv local. Creá uno con:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Red
    Write-Host "  .\.venv\Scripts\pip install -r requirements-mcp-jira.txt" -ForegroundColor Red
    exit 1
}
. $activate

# 6. Correr la demo
Write-Paso "Iniciando RAG + MCP de Jira..."
if ($Pregunta -and $Pregunta.Count -gt 0) {
    python rag_mcp_jira.py @Pregunta
} else {
    python rag_mcp_jira.py
}
