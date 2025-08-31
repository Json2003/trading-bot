param(
    [string] $Entry = 'roman_bot.py',
    [string] $Name = 'RomanBot',
    [string] $Version = '0.1.0'
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

Write-Host "Building release for $Name v$Version"

# Run existing PyInstaller wrapper
if (-not (Test-Path .\build_exe.ps1)) {
    Write-Host "build_exe.ps1 not found in this folder." -ForegroundColor Red
    exit 1
}

& .\build_exe.ps1 -Entry $Entry -Name $Name
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$distExe = Join-Path -Path (Join-Path $here 'dist') -ChildPath "$Name.exe"
if (-not (Test-Path $distExe)) {
    Write-Host "Expected exe not found at $distExe" -ForegroundColor Red
    exit 1
}

$stamp = Get-Date -Format yyyy-MM-dd_HHmmss
$relName = "$Name-$Version-$stamp"
$relDir = Join-Path $here ("releases\$relName")
New-Item -ItemType Directory -Force -Path $relDir | Out-Null

Write-Host "Copying artifacts to $relDir"
Copy-Item -Path $distExe -Destination $relDir -Force

# Include license/readme and model_store if present
foreach ($f in @('..\LICENSE', '..\README.md')) {
    $src = Join-Path $here $f
    if (Test-Path $src) { Copy-Item -Path $src -Destination $relDir -Force }
}

if (Test-Path (Join-Path $here 'model_store')) {
    Copy-Item -Recurse -Path (Join-Path $here 'model_store') -Destination (Join-Path $relDir 'model_store') -Force
}

if (Test-Path (Join-Path $here 'datafiles')) {
    Copy-Item -Recurse -Path (Join-Path $here 'datafiles') -Destination (Join-Path $relDir 'datafiles') -Force
}

$zipPath = Join-Path $here ("releases\$relName.zip")
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Write-Host "Creating ZIP: $zipPath"
Compress-Archive -Path (Join-Path $relDir '*') -DestinationPath $zipPath -Force

Write-Host "ZIP created: $zipPath" -ForegroundColor Green

# Optional: build Inno Setup installer if ISCC is available
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($iscc) {
    Write-Host "ISCC found; building Inno Setup installer"
    $iss = Join-Path $here 'romanbot_installer.iss'
    if (-not (Test-Path $iss)) { Write-Host "Installer script not found: $iss" -ForegroundColor Yellow }
    else {
        & $iscc /Q /O"$(Join-Path $here 'releases')" /F"$Name-Installer" $iss
        if ($LASTEXITCODE -eq 0) { Write-Host "Installer built in releases folder" -ForegroundColor Green }
        else { Write-Host "Installer build failed (ISCC exit $LASTEXITCODE)" -ForegroundColor Red }
    }
} else {
    Write-Host "ISCC (Inno Setup Compiler) not found; skipping installer step." -ForegroundColor Yellow
}

Write-Host "Release complete." -ForegroundColor Cyan
