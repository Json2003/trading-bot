# Setup a Python virtual environment and install project dependencies (PowerShell)
# Usage: Open PowerShell, run with execution policy to allow script if necessary:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#   .\tradingbot_ibkr\scripts\setup_env.ps1

param(
    [string]$venvPath = ".venv",
    [string]$requirements = "tradingbot_ibkr/requirements.txt",
    [string]$pyVersion = "python"
)

Write-Host "Creating virtual environment at" $venvPath
& $pyVersion -m venv $venvPath

Write-Host "Activating virtual environment"
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
} else {
    Write-Error "Activate script not found at $activate"
    exit 1
}

Write-Host "Upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing binary-heavy packages first (numpy, pandas, pyarrow)"
python -m pip install --upgrade pip
python -m pip install numpy==2.2.4 pandas==2.2.2 pyarrow==12.0.0 --prefer-binary

Write-Host "Installing remaining requirements from $requirements"
python -m pip install -r $requirements

Write-Host "Environment setup complete. To activate later: .\$venvPath\Scripts\Activate.ps1"
