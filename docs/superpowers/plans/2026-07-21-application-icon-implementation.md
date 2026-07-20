# 水体反应管检测应用图标 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已批准的“反应管信号”图标实现为统一的 Android 自适应图标、Windows 安装程序图标和官网/GitHub 发布素材。

**Architecture:** 以一份无文字、扁平高对比的图标几何设计作为视觉源，使用内置 imagegen 生成高分辨率视觉稿，再生成项目内的 PNG、ICO 和 Android 自适应图标资源。Android 通过 `mipmap-anydpi-v26` 自适应图标声明，Windows 通过 Inno Setup 的 `SetupIconFile` 接入，官网和发布说明复用同一份 PNG。

**Tech Stack:** Android Gradle Plugin、Android XML resources、PNG/ICO、PowerShell、Inno Setup、Python Pillow、内置 `image_gen`。

---

### Task 1: 先写图标资源契约测试

**Files:**
- Modify: `F:/code/.worktrees/android-water-app/android-app/app/src/test/java/org/hiddenmoon/waterreaction/ReleaseBuildContractTest.java`
- Test command: `F:/code/.worktrees/android-water-app/android-app/gradlew.bat`

- [ ] **Step 1: 在现有发布契约测试中加入失败测试**

在 `ReleaseBuildContractTest` 中加入以下测试。它从现有 `buildFile()` 推导 Android 模块根目录，检查 Manifest 声明和自适应图标资源；当前代码没有这些声明，因此测试应先失败。

```java
@Test
public void launcherIconContractIsDeclared() throws Exception {
    Path appModule = buildFile().getParent();
    Path manifest = appModule.resolve("src/main/AndroidManifest.xml");
    String manifestText = new String(Files.readAllBytes(manifest), StandardCharsets.UTF_8);

    assertTrue(manifestText.contains("android:icon=\"@mipmap/ic_launcher\""));
    assertTrue(manifestText.contains("android:roundIcon=\"@mipmap/ic_launcher_round\""));
    assertTrue(Files.isRegularFile(
            appModule.resolve("src/main/res/mipmap-anydpi-v26/ic_launcher.xml")));
    assertTrue(Files.isRegularFile(
            appModule.resolve("src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml")));
    assertTrue(Files.isRegularFile(
            appModule.resolve("src/main/res/drawable/ic_launcher_foreground.xml")));
}
```

- [ ] **Step 2: 运行失败测试**

运行：

```powershell
Set-Location 'F:\code\.worktrees\android-water-app\android-app'
& '.\gradlew.bat' --no-daemon testDebugUnitTest --tests org.hiddenmoon.waterreaction.ReleaseBuildContractTest.launcherIconContractIsDeclared
```

预期：FAIL，原因是 Manifest 没有 `android:icon`/`android:roundIcon` 或资源文件不存在。

### Task 2: 生成视觉稿并固化图标源文件

**Files:**
- Create: `F:/code/.worktrees/android-water-app/assets/branding/water-reaction-icon.svg`
- Create: `F:/code/.worktrees/android-water-app/assets/branding/water-reaction-icon.png`
- Create: `F:/code/正式发布准备-v1.0.5/图标/water-reaction-icon-concept.png`

- [ ] **Step 1: 用内置 imagegen 生成 1024×1024 视觉稿**

使用内置 `image_gen`，不要使用 CLI fallback。提示词固定为：

```text
Use case: logo-brand
Asset type: application launcher icon
Primary request: original flat icon for a water reaction tube detection application; a single white reaction tube with a bright lime reaction liquid, black measurement marks and a small black detection point, with two simplified cyan/lime scanning signal arcs on the sides
Scene/backdrop: solid signal cyan square background
Style/medium: crisp flat geometric icon, vector-friendly, high contrast, professional laboratory tool
Composition/framing: centered subject with generous safe padding for Android adaptive masks and Windows small icons
Lighting/mood: none; no 3D lighting
Color palette: signal cyan #00A7B5, signal yellow #D9FF3F, ink #111111, paper white #FFFFFF
Text (verbatim): none
Constraints: no text, no letters, no numbers, no logos, no trademarks, no watermark, no gradient, no photo texture, no shadow, no extra objects, keep the tube silhouette readable at 48dp
Avoid: realistic laboratory scene, glass reflections, decorative bubbles, microscopic organisms, complex background
```

将生成结果复制为 `F:\code\正式发布准备-v1.0.5\图标\water-reaction-icon-concept.png`，用 `view_image` 检查主体轮廓、颜色和安全边距。

- [ ] **Step 2: 创建无文字 SVG 源文件**

将同一设计固化到 `assets/branding/water-reaction-icon.svg`，使用 `viewBox="0 0 256 256"`，包含以下几何关系：青色全底；左右两组扫描弧线；白色反应管帽和管身；管身下半部为荧光黄液体；黑色水平刻线和中心定位点。所有主体边界留出至少 36px 安全边距，禁止文字和渐变。

- [ ] **Step 3: 生成统一 PNG 发布稿**

将视觉稿裁切为 1024×1024 正方形，保存为 `assets/branding/water-reaction-icon.png`，并确认 PNG 为 RGBA、尺寸为 1024×1024、四角为主青色背景。正式发布只使用这一份 PNG，不把图标概念稿混入构建产物。

### Task 3: 接入 Android 自适应图标

**Files:**
- Modify: `F:/code/.worktrees/android-water-app/android-app/app/src/main/AndroidManifest.xml`
- Create: `F:/code/.worktrees/android-water-app/android-app/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`
- Create: `F:/code/.worktrees/android-water-app/android-app/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml`
- Create: `F:/code/.worktrees/android-water-app/android-app/app/src/main/res/drawable/ic_launcher_foreground.xml`

- [ ] **Step 1: 添加自适应图标背景和前景**

`ic_launcher.xml` 和 `ic_launcher_round.xml` 使用相同内容，背景引用现有 `@color/archive_signal_cyan`，前景引用 `@drawable/ic_launcher_foreground`：

```xml
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/archive_signal_cyan" />
    <foreground android:drawable="@drawable/ic_launcher_foreground" />
</adaptive-icon>
```

前景 VectorDrawable 使用 `viewportWidth="256"`、`viewportHeight="256"`，将 SVG 中的白色反应管、荧光黄液体、黑色刻线/定位点和两侧扫描弧线转换为 `path`，不引入文字、位图或渐变。

- [ ] **Step 2: 修改 Manifest**

在 `<application>` 上加入：

```xml
android:icon="@mipmap/ic_launcher"
android:roundIcon="@mipmap/ic_launcher_round"
```

保留现有 `android:label`、`android:theme`、Provider 和 Activity 配置不变。

- [ ] **Step 3: 运行契约测试确认通过**

运行：

```powershell
Set-Location 'F:\code\.worktrees\android-water-app\android-app'
& '.\gradlew.bat' --no-daemon testDebugUnitTest --tests org.hiddenmoon.waterreaction.ReleaseBuildContractTest
```

预期：`ReleaseBuildContractTest` 全部 PASS。

### Task 4: 生成 Windows ICO 和发布素材

**Files:**
- Create: `F:/code/.worktrees/android-water-app/packaging/windows/water-reaction-icon.ico`
- Create: `F:/code/.worktrees/android-water-app/scripts/package-icon-assets.py`
- Modify: `F:/code/.worktrees/android-water-app/.gitignore`

- [ ] **Step 1: 写入确定性的 Pillow 转换脚本**

`scripts/package-icon-assets.py` 接受两个位置参数 `source_png` 和 `target_ico`，读取 RGBA PNG，生成包含 16、24、32、48、64、128、256 像素图层的 ICO。脚本必须在源文件不存在、源图不是正方形或源图小于 256×256 时抛出异常，不得静默生成低质量图标。

- [ ] **Step 2: 运行转换并检查图层**

运行：

```powershell
Set-Location 'F:\code\.worktrees\android-water-app'
& 'C:\Users\Muelsyse\anaconda3\python.exe' scripts\package-icon-assets.py assets\branding\water-reaction-icon.png packaging\windows\water-reaction-icon.ico
```

使用 Pillow 重新读取 ICO，确认包含 16、24、32、48、64、128、256 尺寸，并用 `view_image` 检查 256px 预览没有明显锯齿、裁切或水印。

- [ ] **Step 3: 允许公开图标资源进入 Git**

`.gitignore` 只应继续忽略构建输出和临时概念稿；不要用全局 `*.png` 或 `*.ico` 规则屏蔽 `assets/branding/water-reaction-icon.png` 与 `packaging/windows/water-reaction-icon.ico`。概念稿仍保留在 `F:\code\正式发布准备-v1.0.5\图标`，不提交到仓库。

### Task 5: 接入 Inno Setup 和官网发布说明

**Files:**
- Modify: `F:/code/.worktrees/android-water-app/packaging/windows/water-detection.iss`
- Modify: `F:/code/.worktrees/android-water-app/scripts/build-windows-formal-release.ps1`
- Modify: `F:/code/.worktrees/android-water-app/README.md`
- Modify: `F:/code/.worktrees/android-water-app/docs/正式发布指南.md`

- [ ] **Step 1: 为 Inno Setup 添加图标参数**

在 `.iss` 的 `#define` 区域读取 `WATER_WINDOWS_ICON_FILE`，并在 `[Setup]` 中添加：

```iss
#define IconFile GetEnv("WATER_WINDOWS_ICON_FILE")

[Setup]
SetupIconFile={#IconFile}
```

如果 `IconFile` 为空或文件不存在，构建脚本必须在调用 ISCC 前抛出中文错误，不生成无图标安装程序。

- [ ] **Step 2: 让正式构建脚本传入并验证图标**

在调用 ISCC 前设置：

```powershell
$iconFile = Join-Path $repo 'packaging\windows\water-reaction-icon.ico'
if (-not (Test-Path -LiteralPath $iconFile -PathType Leaf)) {
    throw "Windows 安装包缺少图标：$iconFile"
}
$env:WATER_WINDOWS_ICON_FILE = $iconFile
```

只在构建安装包时使用该环境变量；便携 ZIP 仍复用同一份 PNG/ICO 发布素材。

- [ ] **Step 3: 更新发布文档**

在 README 和正式发布指南中加入图标资源路径、Android/Windows 已使用同一视觉稿的说明，以及官网直下素材路径 `assets/branding/water-reaction-icon.png`。不在文档中写入签名密码、服务器 Token 或私钥路径。

### Task 6: 全量验证、提交和交付

**Files:**
- Verify: `F:/code/.worktrees/android-water-app/android-app`
- Verify: `F:/code/.worktrees/android-water-app/assets/branding/water-reaction-icon.png`
- Verify: `F:/code/.worktrees/android-water-app/packaging/windows/water-reaction-icon.ico`

- [ ] **Step 1: 运行 Android 测试和资源构建**

运行：

```powershell
Set-Location 'F:\code\.worktrees\android-water-app\android-app'
& '.\gradlew.bat' --no-daemon testDebugUnitTest assembleDebug
```

预期：测试通过，Android debug APK 成功生成，Manifest 能解析 `@mipmap/ic_launcher` 和 `@mipmap/ic_launcher_round`。

- [ ] **Step 2: 做资源检查**

运行：

```powershell
Set-Location 'F:\code\.worktrees\android-water-app'
& 'C:\Users\Muelsyse\anaconda3\python.exe' -c "from PIL import Image; p=Image.open('assets/branding/water-reaction-icon.png'); assert p.size==(1024,1024); assert p.mode=='RGBA'; print(p.size,p.mode)"
git diff --check
```

预期：PNG 检查输出 `(1024, 1024) RGBA`，Git 空白检查无输出。

- [ ] **Step 3: 检查 Git 范围**

运行：

```powershell
git -C 'F:\code\.worktrees\android-water-app' status --short
git -C 'F:\code\.worktrees\android-water-app' diff --stat
```

只暂存图标实现、测试、Inno/构建脚本、发布文档和公开资源；保留现有 `.superpowers/brainstorm/ui-20260720/` 不变。

- [ ] **Step 4: 提交图标实现**

```powershell
git -C 'F:\code\.worktrees\android-water-app' add -- `
  assets/branding/water-reaction-icon.svg `
  assets/branding/water-reaction-icon.png `
  packaging/windows/water-reaction-icon.ico `
  packaging/windows/water-detection.iss `
  scripts/package-icon-assets.py `
  scripts/build-windows-formal-release.ps1 `
  android-app/app/src/main/AndroidManifest.xml `
  android-app/app/src/main/res/drawable/ic_launcher_foreground.xml `
  android-app/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml `
  android-app/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml `
  android-app/app/src/test/java/org/hiddenmoon/waterreaction/ReleaseBuildContractTest.java `
  README.md docs/正式发布指南.md
git -C 'F:\code\.worktrees\android-water-app' commit -m 'feat: add water reaction application icon'
```

交付时报告 Android 测试结果、资源路径、ICO 尺寸和提交哈希；不报告或复制任何签名密钥和密码。
