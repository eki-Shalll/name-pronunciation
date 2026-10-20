$ErrorActionPreference = "Stop"

Write-Host "Qwen3-TTS Pronunciation Lab" -ForegroundColor Cyan

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("3.12", "3.11", "3.13", "3.10", "3.9")) {
            & py "-$ver" -c "import sys; print(sys.executable)" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return @("py", "-$ver") }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "Python 3.9+ was not found. Install Python first, then run this script again."
}

if (-not (Test-Path ".venv-qwen")) {
    Write-Host "Creating Python virtual environment..."
    $pycmd = Get-PythonCommand
    if ($pycmd.Count -eq 2) { & $pycmd[0] $pycmd[1] -m venv .venv-qwen }
    else { & $pycmd[0] -m venv .venv-qwen }
}

$python = Join-Path $PWD ".venv-qwen\Scripts\python.exe"

Write-Host "Updating pip..."
& $python -m pip install --upgrade pip

Write-Host "Installing/updating Qwen3-TTS dependencies..."
& $python -m pip install -U qwen-tts

Write-Host "Starting local lab at http://127.0.0.1:8765"
Write-Host "The first generation downloads the selected model from Hugging Face." -ForegroundColor Yellow
& $python .\qwen-lab-server.py
