<#
    Demo minima de Ollama + Gemma para clase.
    Uso:
      .\iniciar_gemma_demo.ps1                  -> pregunta por defecto "hola mundo"
      .\iniciar_gemma_demo.ps1 "que es un LLM?" -> pregunta propia
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

# 2. Verificar que el modelo gemma3 este descargado
Write-Paso "Verificando modelo (gemma3)..."
$modelosInstalados = (ollama list) -join "`n"

if ($modelosInstalados -notmatch [regex]::Escape("gemma3")) {
    Write-Host "Descargando gemma3 (puede tardar unos minutos)..." -ForegroundColor Yellow
    ollama pull gemma3
} else {
    Write-Host "  gemma3 ya esta descargado." -ForegroundColor Green
}

# 3. Activar el entorno virtual
Write-Paso "Activando entorno virtual..."
$activate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "No se encontro el venv en .venv. Crea uno con: python -m venv .venv" -ForegroundColor Red
    exit 1
}
. $activate

# 4. Correr el ejemplo
Write-Paso "Iniciando ejemplo de Ollama + Gemma..."
if ($Pregunta -and $Pregunta.Count -gt 0) {
    python ollama_gemma_example.py @Pregunta
} else {
    python ollama_gemma_example.py
}
