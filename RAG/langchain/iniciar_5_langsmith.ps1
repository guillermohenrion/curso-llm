<#
    RAG con LangSmith tracing (LangChain) para clase.
    Uso:
      .\iniciar_5_langsmith.ps1                  -> modo interactivo
      .\iniciar_5_langsmith.ps1 "que es RAG?"    -> una sola pregunta y listo
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

# 2. Verificar que los modelos necesarios estén descargados
Write-Paso "Verificando modelos (nomic-embed-text, gemma3)..."
$modelosInstalados = (ollama list) -join "`n"

foreach ($modelo in @("nomic-embed-text", "gemma3")) {
    if ($modelosInstalados -notmatch [regex]::Escape($modelo)) {
        Write-Host "Descargando $modelo (puede tardar unos minutos)..." -ForegroundColor Yellow
        ollama pull $modelo
    } else {
        Write-Host "  $modelo ya esta descargado." -ForegroundColor Green
    }
}

# 2b. Aviso si falta .env (LangSmith es opcional: sin token igual corre, sin trazas)
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Aviso: no se encontró .env en esta carpeta." -ForegroundColor Yellow
    Write-Host "  Sin el, el script corre igual pero no registra trazas en LangSmith." -ForegroundColor Yellow
    Write-Host "  Para activarlas: copy .env.example .env  y completá LANGSMITH_API_KEY." -ForegroundColor Yellow
}

# 3. Activar el entorno virtual (esta 2 niveles arriba: RAG/langchain/ -> RAG/ -> raiz)
Write-Paso "Activando entorno virtual..."
$RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$activate = Join-Path $RootDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "No se encontró el venv en ..\..\.venv. Creá uno con: python -m venv .venv (en la raiz del proyecto)" -ForegroundColor Red
    exit 1
}
. $activate

# 4. Correr la demo
Write-Paso "Iniciando RAG con LangSmith (LangChain)..."
if ($Pregunta -and $Pregunta.Count -gt 0) {
    python 5_rag_langsmith.py @Pregunta
} else {
    python 5_rag_langsmith.py
}
