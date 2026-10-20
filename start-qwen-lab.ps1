$ErrorActionPreference = "Stop"

Write-Host "Qwen3-TTS Pronunciation Lab" -ForegroundColor Cyan

if (-not (Test-Path ".venv-qwen")) {
    Write-Host "Creating Python virtual environment..."
    py -3.12 -m venv .venv-qwen
}

$python = Join-Path $PWD ".venv-qwen\Scripts\python.exe"
$pip = Join-Path $PWD ".venv-qwen\Scripts\pip.exe"

Write-Host "Updating pip..."
& $python -m pip install --upgrade pip

Write-Host "Installing/updating Qwen3-TTS dependencies..."
& $pip install -U qwen-tts

Write-Host "Starting local lab at http://127.0.0.1:8765"
Write-Host "The first generation downloads the selected model from Hugging Face." -ForegroundColor Yellow
& $python .\qwen-lab-server.py
