[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath,
    [string]$InstallerPath = '',
    [string]$ExpectedVersionName = '1.0.5',
    [string]$Sha256Manifest = '',
    [string]$SignToolPath = '',
    [switch]$RequireCodeSigning
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) { throw "找不到 ZIP：$ZipPath" }
$zip = [IO.Compression.ZipFile]::OpenRead((Get-Item -LiteralPath $ZipPath).FullName)
try {
    $names = @($zip.Entries | ForEach-Object { $_.FullName })
    foreach ($pattern in @(
        '水体反应管检测系统/水体反应管检测系统.exe',
        '水体反应管检测系统/reaction_classifier.h5',
        '水体反应管检测系统/yolov8n.pt',
        '水体反应管检测系统/release.json'
    )) {
        if ($names -notcontains $pattern) { throw "ZIP 缺少关键文件：$pattern" }
    }
    $releaseEntry = $zip.GetEntry('水体反应管检测系统/release.json')
    $releaseText = New-Object IO.StreamReader($releaseEntry.Open())
    try { $release = $releaseText.ReadToEnd() | ConvertFrom-Json } finally { $releaseText.Dispose() }
    if ($release.app_version_name -ne $ExpectedVersionName) { throw 'release.json 版本不匹配' }
    if ($release.api_base_url -notmatch '^https://') { throw 'release.json 未配置 HTTPS 地址' }
    if ($release.app_release_id -notmatch '-desktop$') { throw 'release.json 的 app_release_id 不是桌面发布 ID' }
}
finally {
    $zip.Dispose()
}

if (-not $Sha256Manifest) { $Sha256Manifest = Join-Path (Split-Path -Parent $ZipPath) 'SHA256SUMS.txt' }
if (Test-Path -LiteralPath $Sha256Manifest -PathType Leaf) {
    $zipName = Split-Path -Leaf $ZipPath
    $line = Get-Content -LiteralPath $Sha256Manifest | Where-Object { $_ -match "\s$([regex]::Escape($zipName))$" } | Select-Object -First 1
    if (-not $line) { throw "SHA256SUMS.txt 缺少 $zipName" }
    $expected = ($line -split '\s+')[0].ToUpperInvariant()
    $actual = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($expected -ne $actual) { throw "ZIP SHA256 不匹配：$zipName" }
}

if ($InstallerPath) {
    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) { throw "找不到安装包：$InstallerPath" }
    if (-not $SignToolPath) { $SignToolPath = $env:WATER_SIGNTOOL_PATH }
    if ($RequireCodeSigning) {
        if (-not $SignToolPath -or -not (Test-Path -LiteralPath $SignToolPath -PathType Leaf)) { throw '未提供 signtool.exe' }
        $signature = & $SignToolPath verify /pa /all $InstallerPath 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "安装包 Authenticode 签名校验失败：$signature" }
    } else {
        $signature = Get-AuthenticodeSignature -LiteralPath $InstallerPath
        Write-Warning "安装包签名状态：$($signature.Status)"
    }
}

Write-Host "Windows 正式便携包验证通过：$ZipPath"
