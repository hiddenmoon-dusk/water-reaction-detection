[CmdletBinding()]
param(
    [string]$VersionName = '1.0.5',
    [string]$DistDirectory = '',
    [string]$OutputDirectory = '',
    [string]$ApiBaseUrl = '',
    [string]$BootstrapToken = '',
    [string]$ReleaseBatchId = '',
    [int]$ModelGeneration = 0,
    [int]$DatasetGeneration = 0,
    [string]$InnoSetupPath = '',
    [string]$SignToolPath = '',
    [switch]$RequireInstaller,
    [switch]$RequireCodeSigning
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$projectRoot = [IO.Path]::GetFullPath((Join-Path $repo '..\..'))
if (-not $DistDirectory) {
    $DistDirectory = Join-Path $projectRoot 'dist\水体反应管检测系统'
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot '正式发布准备-v1.0.5\Windows'
}
if (-not $ApiBaseUrl) { $ApiBaseUrl = $env:WATER_PUBLIC_BASE_URL }
if (-not $BootstrapToken) { $BootstrapToken = $env:WATER_BOOTSTRAP_TOKEN }
if (-not $ReleaseBatchId) { $ReleaseBatchId = [Guid]::NewGuid().ToString('N') }

if ($VersionName -notmatch '^\d+\.\d+\.\d+$') { throw 'VersionName 必须使用 x.y.z 格式' }
if ($ReleaseBatchId -notmatch '^[0-9a-f]{32}$') { throw 'ReleaseBatchId 必须是 32 位小写十六进制字符串' }
if (-not $ApiBaseUrl -or $ApiBaseUrl -notmatch '^https://[^\s/]+(?:/[^\s]*)?$') {
    throw '必须通过 -ApiBaseUrl 或 WATER_PUBLIC_BASE_URL 提供 HTTPS 服务器地址'
}
if (-not $BootstrapToken -or $BootstrapToken.Length -lt 16) {
    throw '必须通过 -BootstrapToken 或 WATER_BOOTSTRAP_TOKEN 提供至少 16 个字符的 Token'
}
if (-not (Test-Path -LiteralPath $DistDirectory -PathType Container)) { throw "找不到桌面构建目录：$DistDirectory" }

$mainExe = Get-ChildItem -LiteralPath $DistDirectory -Filter '*.exe' -File |
    Where-Object { $_.Name -eq '水体反应管检测系统.exe' } | Select-Object -First 1
if (-not $mainExe) { throw '桌面构建目录缺少 水体反应管检测系统.exe' }
foreach ($required in @('reaction_classifier.h5', 'yolov8n.pt', 'release.json', '_internal')) {
    if (-not (Test-Path -LiteralPath (Join-Path $DistDirectory $required))) {
        throw "桌面构建目录缺少 $required"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$packageRoot = Join-Path $OutputDirectory '水体反应管检测系统'
if (Test-Path -LiteralPath $packageRoot) { Remove-Item -LiteralPath $packageRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
Get-ChildItem -LiteralPath $DistDirectory -Force |
    Where-Object { $_.Name -ne '结果' } |
    Copy-Item -Destination $packageRoot -Recurse -Force
$userGuide = Join-Path $repo 'docs\中文使用说明.md'
if (-not (Test-Path -LiteralPath $userGuide -PathType Leaf)) {
    throw "缺少中文使用说明：$userGuide"
}
Copy-Item -LiteralPath $userGuide -Destination (Join-Path $packageRoot '中文使用说明.md') -Force

$releaseConfigPath = Join-Path $packageRoot 'release.json'
$releaseConfig = Get-Content -Raw -LiteralPath $releaseConfigPath | ConvertFrom-Json
$manifestModelGeneration = [int]$releaseConfig.model_generation
$manifestDatasetGeneration = [int]$releaseConfig.dataset_generation
if ($ModelGeneration -le 0) { $ModelGeneration = $manifestModelGeneration }
if ($DatasetGeneration -le 0) { $DatasetGeneration = $manifestDatasetGeneration }
if ($ModelGeneration -le 0 -or $DatasetGeneration -le 0) {
    throw '正式发布的 model_generation/dataset_generation 必须大于 0'
}
$releaseConfig | Add-Member -MemberType NoteProperty -Name release_batch_id -Value $ReleaseBatchId -Force
$releaseConfig.app_release_id = "$ReleaseBatchId-desktop"
$releaseConfig.model_generation = $ModelGeneration
$releaseConfig.dataset_generation = $DatasetGeneration
$releaseConfig.api_base_url = $ApiBaseUrl.TrimEnd('/')
$releaseConfig.bootstrap_token = $BootstrapToken
$releaseConfig | Add-Member -MemberType NoteProperty -Name app_version_code -Value 6 -Force
$releaseConfig | Add-Member -MemberType NoteProperty -Name app_version_name -Value $VersionName -Force
$releaseConfig | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $releaseConfigPath -Encoding UTF8

if ($RequireCodeSigning) {
    if (-not $SignToolPath) { $SignToolPath = $env:WATER_SIGNTOOL_PATH }
    $certificatePath = $env:WATER_CODESIGN_PFX
    $certificatePassword = $env:WATER_CODESIGN_PASSWORD
    if (-not $SignToolPath -or -not (Test-Path -LiteralPath $SignToolPath -PathType Leaf)) {
        throw 'RequireCodeSigning 已启用，但没有可用的 signtool.exe'
    }
    if (-not $certificatePath -or -not (Test-Path -LiteralPath $certificatePath -PathType Leaf)) {
        throw 'RequireCodeSigning 已启用，但 WATER_CODESIGN_PFX 不存在'
    }
    if (-not $certificatePassword) { throw 'RequireCodeSigning 已启用，但未设置 WATER_CODESIGN_PASSWORD' }
    & $SignToolPath sign /fd SHA256 /f $certificatePath /p $certificatePassword /tr 'http://timestamp.digicert.com' /td SHA256 $mainExe.FullName
    if ($LASTEXITCODE -ne 0) { throw '桌面 EXE Authenticode 签名失败' }
}

$zipPath = Join-Path $OutputDirectory "water-detection-desktop-v$VersionName-win64.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $packageRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal

if (-not $InnoSetupPath) { $InnoSetupPath = $env:INNO_SETUP_PATH }
if (-not $InnoSetupPath) {
    foreach ($candidate in @(
        'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        'C:\Program Files\Inno Setup 6\ISCC.exe'
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $InnoSetupPath = $candidate; break }
    }
}
$installerPath = Join-Path $OutputDirectory "water-detection-desktop-v$VersionName-setup.exe"
if ($InnoSetupPath) {
    $iconFile = Join-Path $repo 'packaging\windows\water-reaction-icon.ico'
    if (-not (Test-Path -LiteralPath $iconFile -PathType Leaf)) {
        throw "Windows 安装包缺少图标：$iconFile"
    }
    $env:WATER_DESKTOP_VERSION = $VersionName
    $env:WATER_DESKTOP_DIST_DIR = $packageRoot
    $env:WATER_WINDOWS_OUTPUT_DIR = $OutputDirectory
    $env:WATER_WINDOWS_ICON_FILE = $iconFile
    & $InnoSetupPath (Join-Path $repo 'packaging\windows\water-detection.iss')
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup 安装包构建失败' }
    if ($RequireCodeSigning) {
        & $SignToolPath sign /fd SHA256 /f $certificatePath /p $certificatePassword /tr 'http://timestamp.digicert.com' /td SHA256 $installerPath
        if ($LASTEXITCODE -ne 0) { throw '安装包 Authenticode 签名失败' }
    }
} elseif ($RequireInstaller) {
    throw '未找到 Inno Setup 6 的 ISCC.exe，请安装后重新执行，或通过 -InnoSetupPath 指定路径'
} else {
    Write-Warning '未找到 Inno Setup 6；已生成便携 ZIP，但尚未生成安装包。'
}

$metadata = [ordered]@{
    schema_version = 1
    version_name = $VersionName
    app_release_id = "$ReleaseBatchId-desktop"
    model_generation = $ModelGeneration
    dataset_generation = $DatasetGeneration
    api_base_url = $ApiBaseUrl.TrimEnd('/')
    installer = [bool](Test-Path -LiteralPath $installerPath -PathType Leaf)
    code_signed = $RequireCodeSigning.IsPresent
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'release-metadata.json') -Encoding UTF8

$assets = @($zipPath)
if (Test-Path -LiteralPath $installerPath -PathType Leaf) { $assets += $installerPath }
$hashLines = foreach ($asset in $assets) {
    $hash = (Get-FileHash -LiteralPath $asset -Algorithm SHA256).Hash.ToUpperInvariant()
    "$hash  $([IO.Path]::GetFileName($asset))"
}
$hashLines | Set-Content -LiteralPath (Join-Path $OutputDirectory 'SHA256SUMS.txt') -Encoding ASCII
Write-Host "Windows 正式发布资产已生成：$OutputDirectory"
