[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApkPath,
    [int]$ExpectedVersionCode = 6,
    [string]$ExpectedVersionName = '1.0.5',
    [string]$ExpectedPackage = 'org.hiddenmoon.waterreaction',
    [string]$Aapt2Path = '',
    [string]$ApksignerPath = '',
    [string]$Sha256Manifest = ''
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ApkPath -PathType Leaf)) {
    throw "找不到 APK：$ApkPath"
}
$apk = (Get-Item -LiteralPath $ApkPath).FullName

function Resolve-AndroidTool([string]$explicitPath, [string]$name) {
    if ($explicitPath) {
        if (-not (Test-Path -LiteralPath $explicitPath -PathType Leaf)) {
            throw "找不到工具：$explicitPath"
        }
        return (Get-Item -LiteralPath $explicitPath).FullName
    }
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($root in @($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT)) {
        if (-not $root) { continue }
        $candidate = Get-ChildItem -LiteralPath (Join-Path $root 'build-tools') -Filter $name -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    throw "找不到 $name，请通过参数显式指定路径"
}

$aapt2 = Resolve-AndroidTool $Aapt2Path 'aapt2.exe'
$apksigner = Resolve-AndroidTool $ApksignerPath 'apksigner.bat'
if ($apksigner -eq 'apksigner.bat') {
    $apksigner = Resolve-AndroidTool $ApksignerPath 'apksigner.exe'
}

$badging = (& $aapt2 dump badging $apk 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) { throw "aapt2 无法读取 APK：$badging" }
$packageMatch = [regex]::Match($badging, "package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'")
$sdkMatch = [regex]::Match($badging, "sdkVersion:'([^']+)'")
$targetSdkMatch = [regex]::Match($badging, "targetSdkVersion:'([^']+)'")
if (-not $packageMatch.Success) { throw 'APK 缺少可解析的 package badging 信息' }
if ($packageMatch.Groups[1].Value -ne $ExpectedPackage) { throw "包名不匹配：$($packageMatch.Groups[1].Value)" }
if ([int]$packageMatch.Groups[2].Value -ne $ExpectedVersionCode) { throw 'versionCode 不匹配' }
if ($packageMatch.Groups[3].Value -ne $ExpectedVersionName) { throw 'versionName 不匹配' }
if (-not $sdkMatch.Success -or [int]$sdkMatch.Groups[1].Value -ne 34) { throw 'minSdk 不是 34' }
if (-not $targetSdkMatch.Success -or [int]$targetSdkMatch.Groups[1].Value -ne 36) { throw 'targetSdk 不是 36' }

$signature = (& $apksigner verify --verbose --print-certs $apk 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $signature -match 'DOES NOT VERIFY') {
    throw "APK 签名校验失败：$signature"
}
if ($signature -match '(?i)CN=Android Debug|Android Debug') {
    throw 'APK 使用 Debug 证书，不能作为正式发行包'
}
if ($signature -notmatch 'Verified using v2 scheme: true') {
    throw 'APK 未通过 v2 签名校验'
}

if (-not $Sha256Manifest) {
    $Sha256Manifest = Join-Path (Split-Path -Parent $apk) 'SHA256SUMS.txt'
}
if (Test-Path -LiteralPath $Sha256Manifest -PathType Leaf) {
    $fileName = Split-Path -Leaf $apk
    $manifestLine = Get-Content -LiteralPath $Sha256Manifest |
        Where-Object { $_ -match "\s$([regex]::Escape($fileName))$" } |
        Select-Object -First 1
    if (-not $manifestLine) { throw "SHA256SUMS.txt 缺少 $fileName" }
    $expectedHash = ($manifestLine -split '\s+')[0].ToUpperInvariant()
    $actualHash = (Get-FileHash -LiteralPath $apk -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($expectedHash -ne $actualHash) { throw "SHA256 不匹配：$fileName" }
}

Write-Host "Android 正式 APK 验证通过：$apk"
