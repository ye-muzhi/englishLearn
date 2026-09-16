$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root ".tools"
$Uv = Join-Path $Tools "uv.exe"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Url = "http://localhost:8501"

function Test-EnglishLearnReady {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$Url/_stcore/health" -TimeoutSec 1 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Path $Python)) {
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    if (-not (Test-Path $Uv)) {
        Write-Host "Preparing the EnglishLearn installer..."
        $env:UV_UNMANAGED_INSTALL = $Tools
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    Write-Host "Installing EnglishLearn. The first launch can take several minutes..."
    & $Uv venv --allow-existing --python 3.11 (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }
    & $Uv pip install --python $Python -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install EnglishLearn dependencies." }
}

if (Test-EnglishLearnReady) {
    Start-Process $Url
    exit 0
}

$LegacyDataHome = Join-Path $Root "work"
$LegacyProjects = Join-Path $LegacyDataHome "projects.json"
$LegacyModels = Join-Path $LegacyDataHome "models"
$DataHome = Join-Path $env:LOCALAPPDATA "EnglishLearn"
if ((Test-Path $LegacyProjects) -or (Test-Path $LegacyModels)) {
    $DataHome = $LegacyDataHome
}
$env:ENGLISHLEARN_WORK_DIR = $DataHome
$env:PYTHONDONTWRITEBYTECODE = "1"
$Process = Start-Process -FilePath $Python -ArgumentList @(
    "-B", "-m", "streamlit", "run", (Join-Path $Root "app.py"),
    "--server.port", "8501", "--server.fileWatcherType", "none"
) -WorkingDirectory $Root -PassThru -WindowStyle Hidden

for ($Attempt = 0; $Attempt -lt 80; $Attempt++) {
    if (Test-EnglishLearnReady) {
        Start-Process $Url
        Write-Host "EnglishLearn is ready: $Url"
        exit 0
    }
    if ($Process.HasExited) { throw "EnglishLearn stopped during startup." }
    Start-Sleep -Milliseconds 250
}
throw "EnglishLearn did not become ready within 20 seconds."
