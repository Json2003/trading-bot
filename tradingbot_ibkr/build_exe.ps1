<#
Build script for creating a single-file Windows EXE using PyInstaller.

Usage (PowerShell, run from repo root):
    cd tradingbot_ibkr
    .\build_exe.ps1

This script expects PyInstaller to be installed in the active Python environment.
It bundles templates, static files, assets, models, and datafiles into the EXE.
#>

param(
    [string] $Entry = 'roman_bot.py',
    [string] $Name = 'RomanBot'
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

Write-Host "Building $Entry into single EXE (name: $Name)"

# Prefer `pyinstaller` command; if missing, fall back to `python -m PyInstaller`
$pyCmd = Get-Command pyinstaller -ErrorAction SilentlyContinue
$useModule = $false
if (-not $pyCmd) {
    Write-Host "pyinstaller CLI not found; will use 'python -m PyInstaller' if available" -ForegroundColor Yellow
    $useModule = $true
}

# Prepare add-data arguments (Windows syntax is 'SRC;DEST')
$adds = @(
    "templates;templates",
    "static;static",
    "assets;assets",
    "models;models",
    "datafiles;datafiles"
)

# Create --add-data parameters; leave strings as-is
$addArgs = $adds | ForEach-Object { "--add-data=$_" }

# Hidden imports (common pywin32 import) - adjust if build fails
$hidden = @("win32timezone", "win32com", "pkg_resources.py2_warn")
$hiddenArgs = $hidden | ForEach-Object { "--hidden-import=$_" }

# Build command parts; construct --name separately to avoid parsing oddities
$base = @('pyinstaller','--noconfirm','--onefile','--windowed')
$nameArg = "--name=" + $Name
$cmd = $base + @($nameArg) + $addArgs + $hiddenArgs + @($Entry)

Write-Host "Running:" -ForegroundColor Cyan
Write-Host ($cmd -join ' ')

if ($useModule) {
    # call python -m PyInstaller with same args but drop a leading 'pyinstaller' token if present
    $args = @($cmd)
    if ($args.Count -gt 0 -and $args[0] -ieq 'pyinstaller') {
        $args = $args[1..($args.Count - 1)]
    }
    $joined = $args -join ' '
    Write-Host "Executing: python -m PyInstaller $joined"
    & python -m PyInstaller @args
} else {
    & pyinstaller @($cmd)
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build complete. Dist folder: dist\$Name.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
