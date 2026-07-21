# Public README Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the public project README as a Chinese-first product showcase with a short English introduction, verified download links, privacy boundaries, and explicit screenshot/video placeholders.

**Architecture:** Keep the README as one self-contained Markdown document. Put the user-facing story and downloads first, keep build/server details in collapsible sections, and mirror the exact document into the source worktree so the public export and source do not diverge. Do not change application code, release artifacts, or server behavior.

**Tech Stack:** GitHub-flavored Markdown, HTML comments/details blocks, PowerShell, `rg`, Git.

---

### Task 1: Replace the public README with the approved showcase structure

**Files:**
- Modify: `F:/code/.publish-water-reaction-detection-20260720-v2/README.md`

- [ ] **Step 1: Write the Chinese-first product header**

Use the current verified product name and release information. The first screen must include:

```markdown
# 水体反应管检测系统

面向水样反应管照片的离线辅助识别工具，支持 Windows 桌面端和 Android 手机端。

An offline-assisted water-reaction-tube detection tool for Windows and Android.
```

Add links/badges for the live website, the public GitHub repository, the `v1.0.5` Release, and Android API 34+. Do not claim a license that has not been confirmed.

- [ ] **Step 2: Add non-broken media placeholders**

Add a visible “展示素材待补充” table and HTML comments describing exactly what the user should later add:

| 素材 | Suggested location | Required content |
| --- | --- | --- |
| Windows 主界面截图 | `docs/screenshots/01-main-window.png` | Clean UI with no real user data |
| Detection result screenshot | `docs/screenshots/02-detection-result.png` | Boxes, labels, confidence and save action |
| Android flow screenshot | `docs/screenshots/03-android-flow.png` | Photo selection, detection and result states |
| Demo video | GitHub upload or public video URL | 30–90 second photo-to-result flow with no private data or tokens |

Do not add image links to files that do not exist yet, so the public README has no broken media.

- [ ] **Step 3: Add verified capability and usage sections**

Write sections in this order: `核心功能`, `基本使用流程`, `下载与安装`, and `版本与兼容性`. State only verified capabilities: offline model inference, Windows/Android clients, local result saving, optional user-confirmed synchronization, and server release/statistics services. Explain Android 14/API 34+ and Windows x64 portable package requirements.

- [ ] **Step 4: Add the privacy, accuracy and known-limitations sections**

Include the exact current policy facts: images and results are uploaded together when synchronization is used; server data is normally retained for one day; purposes are model optimization and detection-result statistics; deletion requests go to `sunx77@mail2.sysu.edu.cn`; users must remove personal privacy from photos before upload. State that results do not replace laboratory testing, human review, or regulatory conclusions.

Document the current Windows limitation: no Inno Setup installer and no Authenticode signature, so SmartScreen may show “unknown publisher”; the EXE must remain with `_internal`, model files, and `release.json`.

- [ ] **Step 5: Add compact developer sections**

Use `<details>` blocks for `从源码运行`, `项目结构`, `服务器部署提示`, and `安全边界`. Keep formal build commands with safe placeholders such as `https://your-domain.example` and `在本机输入的发布 Token`; never include a password, private key, SSH path, or real bootstrap token.

### Task 2: Mirror the README into the source worktree

**Files:**
- Modify: `F:/code/.worktrees/android-water-app/README.md`

- [ ] **Step 1: Copy the approved public README content exactly**

Use the same section order, wording, media placeholder table, links, limitations, and developer details as Task 1. The two files must be byte-equivalent after line-ending normalization.

- [ ] **Step 2: Check the mirrored files**

Run:

```powershell
Get-FileHash 'F:/code/.publish-water-reaction-detection-20260720-v2/README.md' -Algorithm SHA256
Get-FileHash 'F:/code/.worktrees/android-water-app/README.md' -Algorithm SHA256
```

Expected: identical hashes on the same Windows checkout.

### Task 3: Validate public README quality and links

**Files:**
- Test: `F:/code/.publish-water-reaction-detection-20260720-v2/README.md`
- Test: `F:/code/.worktrees/android-water-app/README.md`

- [ ] **Step 1: Run formatting and stale-content checks**

Run:

```powershell
git diff --check
rg -n 'v1\.0\.4|your-domain\.example|041021sxcyf|BEGIN .*PRIVATE|WATER_ADMIN_INITIAL_PASSWORD=.*[^ ]' README.md
```

Expected: `git diff --check` has no errors and the sensitive/stale-content search has no real-secret matches. Safe example placeholders for the build commands are allowed.

- [ ] **Step 2: Check required public links**

Run a PowerShell request against the website, privacy page, terms page, GitHub repository, and `v1.0.5` Release. Expected: each returns HTTP 200 or a normal GitHub redirect to the public page.

- [ ] **Step 3: Confirm placeholder guidance is discoverable**

Run:

```powershell
rg -n '展示素材待补充|01-main-window|02-detection-result|03-android-flow|30.?90 秒|视频' README.md
```

Expected: all five placeholder cues are present in the public README.

### Task 4: Commit and publish the README update

**Files:**
- Commit: `F:/code/.publish-water-reaction-detection-20260720-v2/README.md`
- Commit: `F:/code/.worktrees/android-water-app/README.md`

- [ ] **Step 1: Commit the public repository README**

Run in the public export repository:

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: refresh public README"
git push origin main
```

Expected: `main` advances on `hiddenmoon-dusk/water-reaction-detection`.

- [ ] **Step 2: Commit the mirrored source README**

Run in the source worktree:

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: mirror public README"
```

Expected: only the README change is committed; existing user-owned `.superpowers/brainstorm/` directories remain untouched.

- [ ] **Step 3: Verify the public result**

Fetch the public raw README or GitHub page and confirm the new Chinese-first header, download links, privacy section, and media placeholder table are visible. Confirm the public repository contains no signing files or credentials.
