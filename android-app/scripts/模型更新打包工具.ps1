Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$android = Join-Path $repo 'android-app'
$python = 'C:\Users\Muelsyse\anaconda3\python.exe'
$javaHome = 'F:\code\.toolchains\jdk17\jdk-17.0.19+10'
$gradleHome = 'F:\code\.gradle-cache'

$form = New-Object Windows.Forms.Form
$form.Text = '水体检测 App 模型更新工具'
$form.Size = New-Object Drawing.Size(720, 420)
$form.StartPosition = 'CenterScreen'
$form.Font = New-Object Drawing.Font('Microsoft YaHei UI', 10)

function Add-FileRow([string]$label, [string]$default, [int]$top, [string]$filter) {
    $caption = New-Object Windows.Forms.Label
    $caption.Text = $label
    $caption.Location = New-Object Drawing.Point(20, $top)
    $caption.Size = New-Object Drawing.Size(100, 32)
    $form.Controls.Add($caption)

    $box = New-Object Windows.Forms.TextBox
    $box.Text = $default
    $box.Location = New-Object Drawing.Point(120, $top)
    $box.Size = New-Object Drawing.Size(470, 32)
    $form.Controls.Add($box)

    $browse = New-Object Windows.Forms.Button
    $browse.Text = '选择'
    $browse.Location = New-Object Drawing.Point(600, $top - 2)
    $browse.Size = New-Object Drawing.Size(80, 34)
    $browse.Add_Click({
        $dialog = New-Object Windows.Forms.OpenFileDialog
        $dialog.Filter = $filter
        if ($dialog.ShowDialog() -eq 'OK') { $box.Text = $dialog.FileName }
    }.GetNewClosure())
    $form.Controls.Add($browse)
    return $box
}

$detectorBox = Add-FileRow '检测模型 (.pt)' (Join-Path $repo 'models\detector.pt') 35 'PyTorch 模型 (*.pt)|*.pt'
$classifierBox = Add-FileRow '分类模型 (.h5)' (Join-Path $repo 'models\classifier.h5') 85 'Keras 模型 (*.h5)|*.h5'

$status = New-Object Windows.Forms.TextBox
$status.Multiline = $true
$status.ReadOnly = $true
$status.ScrollBars = 'Vertical'
$status.Location = New-Object Drawing.Point(20, 145)
$status.Size = New-Object Drawing.Size(660, 160)
$status.Text = "选择两个新模型，然后点击开始。`r`n模型会自动转换、封装进 APK，并复制到 F:\code\发布。"
$form.Controls.Add($status)

$start = New-Object Windows.Forms.Button
$start.Text = '开始转换并生成 APK'
$start.Location = New-Object Drawing.Point(210, 325)
$start.Size = New-Object Drawing.Size(300, 48)
$start.BackColor = [Drawing.Color]::FromArgb(9, 112, 132)
$start.ForeColor = [Drawing.Color]::White
$form.Controls.Add($start)

function Set-Status([string]$message) {
    $status.AppendText("`r`n[$(Get-Date -Format HH:mm:ss)] $message")
    $status.SelectionStart = $status.TextLength
    $status.ScrollToCaret()
    [Windows.Forms.Application]::DoEvents()
}

$start.Add_Click({
    $originalLocation = Get-Location
    try {
        $detector = $detectorBox.Text.Trim()
        $classifier = $classifierBox.Text.Trim()
        if (-not (Test-Path -LiteralPath $detector)) { throw '请选择有效的 .pt 检测模型' }
        if (-not (Test-Path -LiteralPath $classifier)) { throw '请选择有效的 .h5 分类模型' }
        if (-not (Test-Path -LiteralPath $python)) { throw '找不到模型转换 Python 环境' }

        $start.Enabled = $false
        $batch = [Guid]::NewGuid().ToString('N')
        $generated = Join-Path $repo 'model-contract\generated'
        $current = Join-Path $generated 'current'
        $staged = Join-Path $generated $batch
        $versionCode = 1
        $modelGeneration = 1
        if (Test-Path -LiteralPath (Join-Path $current 'model-manifest.json')) {
            $old = Get-Content -Raw -LiteralPath (Join-Path $current 'model-manifest.json') | ConvertFrom-Json
            $versionCode = [int]$old.app_version_code + 1
            $modelGeneration = [int]$old.model_generation + 1
        }
        $versionName = "1.0.$($versionCode - 1)"

        Set-Status '正在转换模型，首次可能需要几分钟…'
        $env:YOLO_AUTOINSTALL = 'false'
        Push-Location $repo
        & $python -m water_models.cli convert `
            --detector $detector --classifier $classifier --output $staged `
            --batch-id $batch --app-release-id "$batch-android" `
            --model-generation $modelGeneration --dataset-generation 1 `
            --version-code $versionCode --version-name $versionName 2>&1 | ForEach-Object { Set-Status $_ }
        if ($LASTEXITCODE -ne 0) { throw "模型转换失败，退出码 $LASTEXITCODE" }

        $resolvedGenerated = [IO.Path]::GetFullPath($generated)
        $resolvedCurrent = [IO.Path]::GetFullPath($current)
        if (-not $resolvedCurrent.StartsWith($resolvedGenerated, [StringComparison]::OrdinalIgnoreCase)) {
            throw '模型输出目录安全校验失败'
        }
        if (Test-Path -LiteralPath $current) { Remove-Item -LiteralPath $current -Recurse -Force }
        Move-Item -LiteralPath $staged -Destination $current

        Set-Status '模型转换完成，正在构建 Android APK…'
        $env:JAVA_HOME = $javaHome
        $env:GRADLE_USER_HOME = $gradleHome
        Push-Location $android
        & .\gradlew.bat --no-daemon clean assembleDebug `
            "-PappVersionCode=$versionCode" "-PappVersionName=$versionName" 2>&1 | ForEach-Object { Set-Status $_ }
        if ($LASTEXITCODE -ne 0) { throw "APK 构建失败，退出码 $LASTEXITCODE" }
        Pop-Location

        $apk = Join-Path $android 'app\build\outputs\apk\debug\app-debug.apk'
        $releaseDir = 'F:\code\发布'
        New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
        $releaseApk = Join-Path $releaseDir "水体反应管检测-v$versionName.apk"
        Copy-Item -LiteralPath $apk -Destination $releaseApk -Force
        Set-Status "完成：$releaseApk"
        [Windows.Forms.MessageBox]::Show("APK 已生成：`r`n$releaseApk", '完成', 'OK', 'Information') | Out-Null
    } catch {
        Set-Status "失败：$($_.Exception.Message)"
        [Windows.Forms.MessageBox]::Show($_.Exception.Message, '打包失败', 'OK', 'Error') | Out-Null
    } finally {
        Set-Location $originalLocation
        $start.Enabled = $true
    }
})

[void]$form.ShowDialog()
