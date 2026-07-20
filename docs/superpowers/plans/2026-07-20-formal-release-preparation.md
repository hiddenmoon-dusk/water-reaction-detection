# 正式发布准备 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将水体反应管检测项目整理为可安全重复执行的正式发布流程，并把所有关键发布模板、清单和待输入项集中到 `F:\code\正式发布准备-v1.0.5`。

**Architecture:** 保留现有 Windows、Android 和 Flask 业务逻辑，仅补充发布边界层：服务器生产配置强制从环境读取，Android Release 构建只接受仓库外签名配置，Windows 使用 Inno Setup 模板生成安装包。发布准备目录保存不含秘密的模板、说明、合规初稿和校验脚本；签名私钥、密码、证书和实际生产地址永远不进入 Git。

**Tech Stack:** Python 3.13 / Flask / Gunicorn / Nginx / systemd, Android Gradle Plugin 8.10.1 / Java 17 / SDK 36, PowerShell, Inno Setup, PyInstaller, GitHub Release。

---

### Task 1: 建立独立的正式发布准备包

**Files:**
- Create: `F:\code\正式发布准备-v1.0.5\README-先看.md`
- Create: `F:\code\正式发布准备-v1.0.5\发布检查清单.md`
- Create: `F:\code\正式发布准备-v1.0.5\发布参数.example.json`
- Create: `F:\code\正式发布准备-v1.0.5\服务器生产环境.example.env`
- Create: `F:\code\正式发布准备-v1.0.5\用户隐私政策-初稿.md`
- Create: `F:\code\正式发布准备-v1.0.5\用户协议-初稿.md`
- Create: `F:\code\正式发布准备-v1.0.5\第三方许可核对表.md`
- Create: `F:\code\正式发布准备-v1.0.5\待用户输入.md`

- [ ] **Step 1: 创建目录结构和不含秘密的模板文件**

  `发布参数.example.json` 固定记录下一正式候选版本 `v1.0.5`、Android `versionCode=6`、平台包名和必须由用户填入的域名/邮箱字段；示例环境文件只包含变量名和说明，不包含真实密码或 Token。

- [ ] **Step 2: 编写发布清单和合规初稿**

  清单覆盖 Android 正式签名、Windows 安装包和代码签名、服务器 HTTPS/备份/监控、许可证、隐私政策、用户协议、版本升级和回滚验证，并明确当前 v1.0.4 Debug APK 不能作为正式发行包。

- [ ] **Step 3: 编写待输入项文件**

  只列出真正无法从代码或本机安全推断的输入：Android 签名密码/keystore、Windows 代码签名证书、生产域名、证书邮箱、服务器登录权限和发布渠道账号；不要求用户把秘密发送到聊天中。

### Task 2: 加固服务器生产配置和部署脚本

**Files:**
- Modify: `server/water_server/config.py`
- Modify: `server/deploy/bootstrap.sh`
- Modify: `server/deploy/nginx.conf`
- Modify: `server/deploy/water-detection.service`
- Modify: `server/tests/test_deploy_files.py`
- Modify: `server/tests/test_public.py`
- Create: `server/tests/test_config_security.py`

- [ ] **Step 1: 为生产模式写失败测试**

  增加测试，验证 `WATER_PRODUCTION=true` 时缺少 `WATER_SECRET_KEY`、`WATER_ADMIN_INITIAL_PASSWORD`、`WATER_BOOTSTRAP_TOKEN` 或使用 HTTP 公网地址会被拒绝；测试模式仍使用现有测试密码和 Token，不影响回归测试。

- [ ] **Step 2: 实现生产配置校验**

  `default_config()` 新增 `PRODUCTION` 配置；生产模式从环境变量读取全部密钥，拒绝 `development-change-me`、`change-me-before-use`、空值和默认 bootstrap Token，并要求 `PUBLIC_BASE_URL` 使用 HTTPS。开发模式保留明确的非生产占位值。

- [ ] **Step 3: 参数化部署域名和 HTTPS 证书**

  `bootstrap.sh` 强制读取 `WATER_PUBLIC_DOMAIN`、`WATER_CERTBOT_EMAIL`、`WATER_ADMIN_INITIAL_PASSWORD`、`WATER_BOOTSTRAP_TOKEN`，生成 `WATER_PRODUCTION=true` 的受限 `secrets.env`；Nginx 模板使用安全校验后的域名；Certbot 使用真实邮箱并在证书申请失败时终止部署，不再静默保留 HTTP。

- [ ] **Step 4: 增加生产反向代理安全头和服务约束**

  Nginx 增加 HTTPS 迁移所需的安全头、请求超时和上传限制；systemd 保留 `NoNewPrivileges`、`PrivateTmp`、`ProtectSystem` 并补充服务失败后的清晰健康检查命令。

- [ ] **Step 5: 运行服务器定向测试**

  Run: `C:\Users\Muelsyse\anaconda3\python.exe -m pytest server\tests\test_config_security.py server\tests\test_deploy_files.py server\tests\test_public.py -q`

  Expected: all tests pass。

### Task 3: 增加 Android 正式签名和可复现构建

**Files:**
- Modify: `android-app/app/build.gradle`
- Modify: `.gitignore`
- Create: `android-app/release-signing.properties.example`
- Create: `android-app/scripts/build-formal-release.ps1`
- Create: `android-app/scripts/verify-formal-release.ps1`
- Create: `android-app/app/src/test/java/org/hiddenmoon/waterreaction/ReleaseBuildContractTest.java`

- [ ] **Step 1: 增加 Release 签名契约测试**

  测试通过读取 Gradle 配置文本验证 Release 构建声明 `release-signing.properties`、不使用 debug 签名，并保留包名、minSdk 34、targetSdk 36。

- [ ] **Step 2: 实现仓库外签名配置**

  `app/build.gradle` 从 `android-app/release-signing.properties` 读取 `storeFile`、`storePassword`、`keyAlias`、`keyPassword`；缺少任一项时 `assembleRelease`/`bundleRelease` 直接报出中文错误。真实配置路径和密钥由 `.gitignore` 排除，示例文件不含秘密。

- [ ] **Step 3: 编写正式构建脚本**

  `build-formal-release.ps1` 接受版本号、版本名称、输出目录和模型生成目录，先执行 `testDebugUnitTest`，再执行 `bundleRelease` 与 `assembleRelease`，复制为 ASCII 文件名并生成 `SHA256SUMS.txt`；脚本在签名配置、模型资产或 Java 17 缺失时立即停止。

- [ ] **Step 4: 编写正式产物验证脚本**

  `verify-formal-release.ps1` 使用 `aapt2`/`apksigner` 检查包名、版本、minSdk、targetSdk、签名证书和 SHA256，并拒绝 Debug 证书或校验清单不匹配。

- [ ] **Step 5: 运行 Android 单元测试和无密钥契约检查**

  Run: `cd android-app; .\gradlew.bat --no-daemon testDebugUnitTest`

  Expected: `BUILD SUCCESSFUL`；没有签名配置时只验证 debug 单元测试和构建契约，不伪造正式签名。

### Task 4: 增加 Windows 正式安装包流程

**Files:**
- Create: `packaging/windows/water-detection.iss`
- Create: `scripts/build-windows-formal-release.ps1`
- Create: `scripts/verify-windows-formal-release.ps1`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: 编写 Inno Setup 安装包模板**

  模板从环境变量读取版本和当前 `dist` 目录，安装完整 PyInstaller 目录、模型和 `release.json`，创建开始菜单/桌面快捷方式、卸载程序和安装目录权限说明；默认输出 ASCII 文件名。

- [ ] **Step 2: 编写 Windows 构建脚本**

  `build-windows-formal-release.ps1` 检查 PyInstaller 输出中的 EXE、`_internal`、两个模型和 `release.json`，生成便携 ZIP；发现 Inno Setup 时构建安装包，未安装时给出明确安装路径而不是伪造成功。

- [ ] **Step 3: 编写 Windows 验证脚本**

  验证 ZIP 内关键文件、版本配置、SHA256 和安装包输出；如果提供 `signtool.exe` 与证书路径，验证 Authenticode 签名，否则标记“未签名”并返回非零退出码供正式 CI 使用。

- [ ] **Step 4: 更新中文发布文档**

  README 和中文使用说明增加正式安装包、代码签名、干净 Windows 10/11 验收、升级和回滚要求，并明确当前 v1.0.4 仅是验证发布。

### Task 5: 汇总用户输入并执行可完成的验证

**Files:**
- Modify: `F:\code\正式发布准备-v1.0.5\待用户输入.md`
- Create: `F:\code\正式发布准备-v1.0.5\验证记录.md`

- [ ] **Step 1: 建立验证记录**

  记录已完成的 Python/服务器、模型工具、Android 单元测试，以及因缺少签名、证书或生产域名而暂缓的命令；不记录任何秘密值。

- [ ] **Step 2: 运行全量回归和静态敏感信息扫描**

  Run: `C:\Users\Muelsyse\anaconda3\python.exe -m pytest tests server\tests -q`

  Run: `rg -n --hidden -g '!build/**' -g '!.gradle/**' -g '!*.pyc' 'WATER_ADMIN_INITIAL_PASSWORD=|BEGIN (RSA|EC|OPENSSH) PRIVATE KEY|password=.{0,4}[^<]' .`

  Expected: tests pass；扫描只命中测试用例或示例说明，不命中生产秘密。

- [ ] **Step 3: 将发布准备包和代码改动同步到 GitHub 发布分支**

  只提交源码、模板、脚本和不含秘密的文档；正式 APK、Windows 安装包、签名证书和生产环境文件只在用户补齐输入后生成并作为 Release 资产上传。

---

## 用户需要提供的输入

执行到正式签名和生产部署时才需要用户输入以下内容：

1. Android 正式签名 keystore 路径，或授权生成一个新的 keystore；密码建议在本机终端输入，不要发到聊天。
2. Windows 代码签名证书（PFX）路径及密码，或暂时接受安装包显示“未知发布者”。
3. 生产域名、Certbot 证书邮箱和服务器登录/部署权限。
4. 是否公开 GitHub 仓库、上架 Google Play/其他应用商店，或仅向指定用户分发。

在这些输入到位前，所有不依赖秘密的安全加固、模板、文档、构建检查和验证脚本都可以先完成。
