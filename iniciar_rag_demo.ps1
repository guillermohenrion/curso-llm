<#
    Demo de RAG simple para clase.
    Uso:
      .\iniciar_rag_demo.ps1                  -> modo interactivo (pregunta y pregunta)
      .\iniciar_rag_demo.ps1 "que es RAG?"    -> una sola pregunta y listo
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
        Write-Host "No se pudo iniciar Ollama. Instalalo desde https://ollama.com o iniciá 'ollama serve' manualmente." -ForegroundColor Red
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

# 3. Activar el entorno virtual
Write-Paso "Activando entorno virtual..."
$activate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "No se encontró el venv en .venv. Creá uno con: python -m venv .venv" -ForegroundColor Red
    exit 1
}
. $activate

# 4. Correr la demo
Write-Paso "Iniciando RAG simple..."
if ($Pregunta -and $Pregunta.Count -gt 0) {
    python rag_simple.py @Pregunta
} else {
    python rag_simple.py
}
