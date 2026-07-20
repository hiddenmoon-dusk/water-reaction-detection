# GitHub Release 发布整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将包含 Windows 客户端、Android 客户端源码、模型、服务器和中文说明的干净项目首发到私有 GitHub 仓库，并用 Release 资产分发 EXE、APK 和完整桌面包。

**Architecture:** 以包含 Android 工程的 `feature/android-water-app` 工作树为源，建立不带旧 Git 历史的干净导出目录。源码与可审查文本进入 `main`，构建产物通过 GitHub Release 资产上传；发布前执行凭据扫描、测试和 SHA-256 校验。

**Tech Stack:** Git, GitHub REST API, Python 3.13, PyQt5, TensorFlow, Ultralytics, Flask, Java 17, Android SDK 36, Gradle 8.11.1。

---

### Task 1: 生成干净源代码副本

**Files:**
- Source: `F:/code/.worktrees/android-water-app/`
- Export: `F:/code/.publish-water-reaction-detection-20260720-v2/`
- Exclude: `.superpowers/`, `docs/superpowers/` 原会话资料、`服务器/`、个人文档、构建目录和发布产物

- [x] **Step 1: 复制功能分支的已跟踪源码**

```powershell
git -C F:\code\.worktrees\android-water-app ls-files
```

只复制源码、模型、测试、部署文件和普通项目文档；不复制服务器登录资料、缓存和构建目录。

- [x] **Step 2: 增加发布仓库的忽略规则**

忽略 `服务器/`、`发布/`、`*.download` 和发布暂存目录，继续忽略 APK、EXE、ZIP、数据库、签名文件和模型转换缓存。

### Task 2: 整理说明文档和部署安全

**Files:**
- Create: `README.md`
- Create: `docs/中文使用说明.md`
- Modify: `server/deploy/bootstrap.sh`
- Modify: `server/water_server/config.py`
- Modify: `server/tests/`

- [x] **Step 1: 写入中文使用说明**

说明 Windows 和 Android 的安装、检测模式、结果保存、网络同步、系统版本要求和下载校验方式。

- [x] **Step 2: 删除部署脚本中的硬编码初始管理员密码**

首次部署要求通过 `WATER_ADMIN_INITIAL_PASSWORD` 环境变量传入密码；测试使用独立的非生产测试密码。

### Task 3: 构建产物和校验清单

**Files:**
- Source artifact: `F:/code/dist/水体反应管检测系统/水体反应管检测系统.exe`
- Android artifact: `F:/code/发布/水体反应管检测-v1.0.4.apk`
- Desktop bundle: `F:/code/output/water-detection-desktop.zip`
- Release manifest: `SHA256SUMS.txt`

- [x] **Step 1: 计算每个发布资产的 SHA-256**

```powershell
Get-FileHash <asset> -Algorithm SHA256
```

- [x] **Step 2: 将 EXE、APK、完整桌面 ZIP 和校验清单上传到同一 Release**

源码提交不包含构建产物，下载者从 Release 获取对应版本。

### Task 4: GitHub 发布验证

**Files:**
- Repository: `hiddenmoon-dusk/water-reaction-detection`

- [ ] **Step 1: 验证远程提交树**

通过 GitHub API 回读 `main` 的提交和关键文件，确认 README、中文说明、Android 工程、模型工具、服务器源码和测试均存在。

- [ ] **Step 2: 验证 Release 资产**

回读 Release 的资产名称、文件大小和下载地址，并核对本地 SHA-256 清单。
