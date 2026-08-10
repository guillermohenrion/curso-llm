<#
    Demo A2A (Agent2Agent) con el SDK oficial (a2a-sdk): levanta
    agente_servidor.py en una ventana aparte (si no esta corriendo ya),
    espera a que responda, y le manda un mensaje con cliente.py.

    Uso:
      .\iniciar_a2a_demo.ps1                         -> mensaje de ejemplo
      .\iniciar_a2a_demo.ps1 "hola mundo gracias"    -> tu propio texto
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Texto
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$AgentCardUrl = "http://127.0.0.1:9000/.well-known/agent-card.json"

function Write-Paso($msg) {
    Write-Host ""
    Write-Host ">> $msg" -ForegroundColor Cyan
}

function Test-Servidor {
    try {
        Invoke-WebRequest -Uri $AgentCardUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

# 1. Activar el entorno virtual (esta 2 niveles arriba: a2a-demo/ -> orquestacion/ -> raiz)
Write-Paso "Activando entorno virtual..."
$RootDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$activate = Join-Path $RootDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "No se encontró el venv en ..\..\.venv. Creá uno con: python -m venv .venv (en la raiz del proyecto)" -ForegroundColor Red
    exit 1
}
. $activate

# 2. Verificar que a2a-sdk este instalado
Write-Paso "Verificando dependencias (a2a-sdk)..."
python -c "import a2a" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Falta a2a-sdk. Instalalo con:" -ForegroundColor Red
    Write-Host "  pip install -r requirements-a2a.txt" -ForegroundColor Red
    exit 1
}
Write-Host "  a2a-sdk OK." -ForegroundColor Green

# 3. Reutilizar el servidor si ya esta corriendo (de una corrida anterior)
Write-Paso "Verificando si el servidor A2A ya esta corriendo..."
$servidorPropio = $false
if (Test-Servidor) {
    Write-Host "  Ya esta corriendo en $AgentCardUrl, lo reutilizo." -ForegroundColor Green
} else {
    Write-Paso "Iniciando agente_servidor.py en una ventana nueva..."
    Start-Process -FilePath "python" -ArgumentList "agente_servidor.py" `
        -WorkingDirectory $ScriptDir -WindowStyle Normal
    $servidorPropio = $true

    $intentos = 0
    $listo = $false
    while (-not $listo -and $intentos -lt 15) {
        Start-Sleep -Milliseconds 500
        $intentos++
        $listo = Test-Servidor
    }
    if (-not $listo) {
        Write-Host "No se pudo levantar el servidor A2A (revisa la ventana nueva)." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Servidor OK." -ForegroundColor Green
}

# 4. Correr el cliente contra el servidor
Write-Paso "Ejecutando cliente.py..."
if ($Texto -and $Texto.Count -gt 0) {
    python cliente.py @Texto
} else {
    python cliente.py
}

if ($servidorPropio) {
    Write-Host ""
    Write-Host "El servidor A2A sigue corriendo en la otra ventana." -ForegroundColor Yellow
    Write-Host "Cerrala cuando termines, o corre otra vez este script con otro texto." -ForegroundColor Yellow
}
