# Build a standalone MSP-Control distribution on Windows.
# Run from PowerShell: .\build_windows.ps1
# For a GUI-only build: .\build_windows.ps1 -Windowed

param(
    [switch]$Windowed
)

$ErrorActionPreference = "Stop"

# This script lives in the repository root.
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$configSource = Join-Path $root "config"
$entryPoint = Join-Path $root "src\msp_control\gui\app.py"

foreach ($required in @($python, $configSource, $entryPoint)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path does not exist: $required"
    }
}

# Make a new distribution for each build. Never overwrite the existing
# dist/ output or a deployed copy with user-modified configuration.
$buildTag = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$distParent = Join-Path $root "dist\build-$buildTag"
$releaseDir = Join-Path $distParent "MSP-Control"

$windowMode = if ($Windowed) { "--windowed" } else { "--console" }

$pyinstallerArgs = @(
    "--onedir"
    $windowMode
    "--clean"
    "--noconfirm"
    "--name", "MSP-Control"
    "--paths", (Join-Path $root "src")
    "--collect-data", "msp_control.calibration"
    "--recursive-copy-metadata", "nidaqmx"
    "--distpath", $distParent
    "--workpath", (Join-Path $root "build\pyinstaller")
    "--specpath", (Join-Path $root "build")
    $entryPoint
)

Write-Host "Building MSP-Control ($windowMode)..."
& $python -m PyInstaller @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath (Join-Path $releaseDir "MSP-Control.exe"))) {
    throw "Expected executable not found in $releaseDir"
}

# Keep configuration editable beside the executable, not inside _internal.
# This is a fresh snapshot from the source config; existing distributions
# and their edited configurations are left untouched.
Copy-Item -LiteralPath $configSource -Destination $releaseDir -Recurse

# Git does not store empty directories.
foreach ($name in @("protocols", "calibrations")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $releaseDir "config\$name") | Out-Null
}

$photodiodeCsv = Join-Path $releaseDir "_internal\msp_control\calibration\data\PDGrasby.csv"
if (-not (Test-Path -LiteralPath $photodiodeCsv)) {
    throw "Packaged photodiode responsivity file is missing: $photodiodeCsv"
}

Write-Host ""
Write-Host "Build complete: $releaseDir"
Write-Host "Configuration: $(Join-Path $releaseDir 'config')"
Write-Host "This is a new build; previous distributions have not been changed."
Write-Host "Review the copied calibrations before distributing to another computer."
