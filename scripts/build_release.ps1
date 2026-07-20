param(
    [string]$Python = "C:\Users\Muelsyse\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $ProjectRoot "dist"
$BuildRoot = Join-Path $ProjectRoot "build"
$Release = Join-Path $DistRoot "水体反应管检测系统"
$Output = Join-Path $ProjectRoot "output"

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
if (Test-Path -LiteralPath $Release) {
    Remove-Item -LiteralPath $Release -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $DistRoot, $Output | Out-Null

Push-Location $ProjectRoot
try {
    $env:TF_ENABLE_ONEDNN_OPTS = "0"
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    & $Python -m PyInstaller --noconfirm --clean "water_detection.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Copy-Item -LiteralPath "models\classifier.h5" -Destination (Join-Path $Release "reaction_classifier.h5")
    Copy-Item -LiteralPath "models\detector.pt" -Destination (Join-Path $Release "yolov8n.pt")
    Copy-Item -LiteralPath "release.json" -Destination (Join-Path $Release "release.json")

    $ZipPath = Join-Path $Output "water-detection-desktop.zip"
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Compress-Archive -Path (Join-Path $Release "*") -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Output "Release directory: $Release"
    Write-Output "Release archive: $ZipPath"
}
finally {
    Pop-Location
}
