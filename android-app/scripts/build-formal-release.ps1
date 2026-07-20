[CmdletBinding()]
param(
    [int]$VersionCode = 6,
    [string]$VersionName = '1.0.5',
    [string]$OutputDirectory = '',
    [string]$ModelDirectory = '',
    [string]$ApiBaseUrl = '',
    [string]$BootstrapToken = '',
    [string]$ReleaseBatchId = '',
    [int]$ModelGeneration = 0,
    [int]$DatasetGeneration = 0
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$android = Join-Path $repo 'android-app'
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo '..\正式发布准备-v1.0.5\Android'
}
if (-not $ModelDirectory) {
    $ModelDirectory = Join-Path $repo 'model-contract\generated\current'
}
if (-not $ApiBaseUrl) {
    $ApiBaseUrl = $env:WATER_PUBLIC_BASE_URL
}
if (-not $BootstrapToken) {
    $BootstrapToken = $env:WATER_BOOTSTRAP_TOKEN
}
if (-not $ReleaseBatchId) {
    $ReleaseBatchId = [Guid]::NewGuid().ToString('N')
}

if ($VersionCode -le 0) { throw 'VersionCode 必须大于 0' }
if ($VersionName -notmatch '^\d+\.\d+\.\d+$') {
    throw 'VersionName 必须使用 x.y.z 格式'
}
if ($ReleaseBatchId -notmatch '^[0-9a-f]{32}$') {
    throw 'ReleaseBatchId 必须是 32 位小写十六进制字符串'
}
if (-not $ApiBaseUrl -or $ApiBaseUrl -notmatch '^https://[^\s/]+(?:/[^\s]*)?$') {
    throw '必须通过 -ApiBaseUrl 或 WATER_PUBLIC_BASE_URL 提供 HTTPS 服务器地址'
}
if (-not $BootstrapToken -or $BootstrapToken.Length -lt 16) {
    throw '必须通过 -BootstrapToken 或 WATER_BOOTSTRAP_TOKEN 提供至少 16 个字符的 Token'
}

$signingProperties = Join-Path $android 'release-signing.properties'
if (-not (Test-Path -LiteralPath $signingProperties -PathType Leaf)) {
    throw "找不到正式签名配置：$signingProperties。请复制 release-signing.properties.example 并在仓库外填写。"
}

$manifestPath = Join-Path $ModelDirectory 'model-manifest.json'
foreach ($required in @('detector.tflite', 'classifier.tflite', 'model-manifest.json')) {
    $requiredPath = Join-Path $ModelDirectory $required
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "模型发布目录缺少 $required：$ModelDirectory"
    }
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$manifestModelGeneration = [int]$manifest.model_generation
$manifestDatasetGeneration = [int]$manifest.dataset_generation
if ($ModelGeneration -le 0) { $ModelGeneration = $manifestModelGeneration }
if ($DatasetGeneration -le 0) { $DatasetGeneration = $manifestDatasetGeneration }
if ($ModelGeneration -le 0 -or $DatasetGeneration -le 0) {
    throw '正式发布的 model_generation/dataset_generation 必须大于 0'
}

$javaHome = $env:JAVA_HOME
if (-not $javaHome) { $javaHome = 'F:\code\.toolchains\jdk17\jdk-17.0.19+10' }
if (-not (Test-Path -LiteralPath $javaHome -PathType Container)) {
    throw "找不到 Java 17：$javaHome"
}
$env:JAVA_HOME = $javaHome
$env:WATER_PUBLIC_BASE_URL = $ApiBaseUrl.TrimEnd('/')
$env:WATER_BOOTSTRAP_TOKEN = $BootstrapToken

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$gradleArgs = @(
    '--no-daemon',
    'clean',
    'bundleRelease',
    'assembleRelease',
    '-PformalRelease=true',
    "-PformalAppReleaseId=$ReleaseBatchId-android",
    "-PformalModelGeneration=$ModelGeneration",
    "-PformalDatasetGeneration=$DatasetGeneration",
    "-PappVersionCode=$VersionCode",
    "-PappVersionName=$VersionName"
)

Push-Location $android
try {
    & '.\gradlew.bat' @gradleArgs
    if ($LASTEXITCODE -ne 0) { throw "Android 正式构建失败，退出码 $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$apk = Join-Path $android 'app\build\outputs\apk\release\app-release.apk'
$aab = Join-Path $android 'app\build\outputs\bundle\release\app-release.aab'
if (-not (Test-Path -LiteralPath $apk -PathType Leaf)) { throw "未生成 APK：$apk" }
if (-not (Test-Path -LiteralPath $aab -PathType Leaf)) { throw "未生成 AAB：$aab" }

$apkName = "water-reaction-android-v$VersionName.apk"
$aabName = "water-reaction-android-v$VersionName.aab"
Copy-Item -LiteralPath $apk -Destination (Join-Path $OutputDirectory $apkName) -Force
Copy-Item -LiteralPath $aab -Destination (Join-Path $OutputDirectory $aabName) -Force
$metadata = [ordered]@{
    schema_version = 1
    version_name = $VersionName
    version_code = $VersionCode
    application_id = 'org.hiddenmoon.waterreaction'
    app_release_id = "$ReleaseBatchId-android"
    model_generation = $ModelGeneration
    dataset_generation = $DatasetGeneration
    api_base_url = $ApiBaseUrl.TrimEnd('/')
    signing_configuration = 'android-app/release-signing.properties'
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'release-metadata.json') -Encoding UTF8

$hashLines = foreach ($asset in @($apkName, $aabName)) {
    $hash = (Get-FileHash -LiteralPath (Join-Path $OutputDirectory $asset) -Algorithm SHA256).Hash.ToUpperInvariant()
    "$hash  $asset"
}
$hashLines | Set-Content -LiteralPath (Join-Path $OutputDirectory 'SHA256SUMS.txt') -Encoding ASCII
Write-Host "Android 正式产物已生成：$OutputDirectory"
