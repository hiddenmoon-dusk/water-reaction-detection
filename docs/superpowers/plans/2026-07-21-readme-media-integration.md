# README Media Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy the four user-provided showcase assets into the repository, replace README media placeholders with real screenshots and a clickable demo video, mirror the result into the source worktree, and publish the update.

**Architecture:** Store screenshots under `docs/screenshots/` and the MP4 under `docs/media/` using stable ASCII filenames. The README will render the three screenshots with bounded HTML widths and use the detection-result screenshot as a clickable video cover, avoiding a broken or unsupported inline video player on GitHub.

**Tech Stack:** GitHub-flavored Markdown, HTML `img`/link markup, PowerShell file copy, `ffprobe`, Git.

---

### Task 1: Copy and verify showcase assets in both workspaces

**Files:**
- Create: `F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/01-main-window.png`
- Create: `F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/02-detection-result.jpg`
- Create: `F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/03-android-flow.jpg`
- Create: `F:/code/.publish-water-reaction-detection-20260720-v2/docs/media/demo-v1.0.5.mp4`
- Create: `F:/code/.worktrees/android-water-app/docs/screenshots/01-main-window.png`
- Create: `F:/code/.worktrees/android-water-app/docs/screenshots/02-detection-result.jpg`
- Create: `F:/code/.worktrees/android-water-app/docs/screenshots/03-android-flow.jpg`
- Create: `F:/code/.worktrees/android-water-app/docs/media/demo-v1.0.5.mp4`

- [ ] **Step 1: Create only the two target directories in the public workspace**

Run:

~~~powershell
New-Item -ItemType Directory -Force -Path `
  'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots', `
  'F:/code/.publish-water-reaction-detection-20260720-v2/docs/media' | Out-Null
~~~

Expected: both directories exist; no source files are changed.

- [ ] **Step 2: Copy the four named source assets to stable public paths**

Run:

~~~powershell
Copy-Item -LiteralPath 'F:/code/展示素材/Windows 主界面截图.png' -Destination 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/01-main-window.png'
Copy-Item -LiteralPath 'F:/code/展示素材/检测结果截图.jpg' -Destination 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/02-detection-result.jpg'
Copy-Item -LiteralPath 'F:/code/展示素材/Android 流程截图.jpg' -Destination 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/03-android-flow.jpg'
Copy-Item -LiteralPath 'F:/code/展示素材/演示视频.mp4' -Destination 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/media/demo-v1.0.5.mp4'
~~~

Expected: the four destination files exist and their byte lengths match the source files.

- [ ] **Step 3: Mirror the four assets to the source worktree**

Run:

~~~powershell
New-Item -ItemType Directory -Force -Path `
  'F:/code/.worktrees/android-water-app/docs/screenshots', `
  'F:/code/.worktrees/android-water-app/docs/media' | Out-Null
Copy-Item -LiteralPath 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/01-main-window.png' -Destination 'F:/code/.worktrees/android-water-app/docs/screenshots/01-main-window.png'
Copy-Item -LiteralPath 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/02-detection-result.jpg' -Destination 'F:/code/.worktrees/android-water-app/docs/screenshots/02-detection-result.jpg'
Copy-Item -LiteralPath 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/screenshots/03-android-flow.jpg' -Destination 'F:/code/.worktrees/android-water-app/docs/screenshots/03-android-flow.jpg'
Copy-Item -LiteralPath 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/media/demo-v1.0.5.mp4' -Destination 'F:/code/.worktrees/android-water-app/docs/media/demo-v1.0.5.mp4'
~~~

Expected: each mirrored file has the same SHA-256 as its public-workspace counterpart.

- [ ] **Step 4: Verify the video format before editing README**

Run:

~~~powershell
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate -show_entries format=duration,size -of default=noprint_wrappers=1 'F:/code/.publish-water-reaction-detection-20260720-v2/docs/media/demo-v1.0.5.mp4'
~~~

Expected: H.264 video, 1280×720, 30 fps, approximately 43.5 seconds, and a size below GitHub's 100 MB single-file limit.

### Task 2: Replace README placeholders with real media

**Files:**
- Modify: `F:/code/.publish-water-reaction-detection-20260720-v2/README.md`
- Modify: `F:/code/.worktrees/android-water-app/README.md`

- [ ] **Step 1: Replace the placeholder section in the public README**

Remove the `展示素材待补充` table and its HTML placeholder comments. Insert this section after the introductory product statement:

~~~markdown
## 产品展示

### Windows 主界面

<p align="center">
  <img src="docs/screenshots/01-main-window.png" width="720" alt="Windows 主界面">
</p>

### 检测结果

<p align="center">
  <img src="docs/screenshots/02-detection-result.jpg" width="720" alt="检测结果界面">
</p>

### Android 使用流程

<p align="center">
  <img src="docs/screenshots/03-android-flow.jpg" width="900" alt="Android 使用流程">
</p>

### 操作演示视频

点击下方截图打开约 43 秒的完整演示视频：

<p align="center">
  <a href="docs/media/demo-v1.0.5.mp4">
    <img src="docs/screenshots/02-detection-result.jpg" width="720" alt="点击查看操作演示视频">
  </a>
</p>

> 展示素材使用演示数据制作；实际上传照片前，请确认其中不含个人隐私。
~~~

- [ ] **Step 2: Copy the approved README content to the source workspace**

Make the source README byte-equivalent to the public README after line-ending normalization. Do not leave the old placeholder filenames in either README.

### Task 3: Validate links, media and repository safety

**Files:**
- Test: both README files and the eight mirrored media paths.

- [ ] **Step 1: Verify every media path and mirror hash**

Run:

~~~powershell
$paths=@(
  'docs/screenshots/01-main-window.png',
  'docs/screenshots/02-detection-result.jpg',
  'docs/screenshots/03-android-flow.jpg',
  'docs/media/demo-v1.0.5.mp4'
)
foreach($path in $paths){if(!(Test-Path -LiteralPath $path)){throw "missing=$path"}}
~~~

Expected: no `missing=` output and all four public/source pairs have equal SHA-256 values.

- [ ] **Step 2: Validate Markdown and placeholder removal**

Run in the public workspace:

~~~powershell
git diff --check
if((rg -n '展示素材待补充|01-main-window\.png.*建议|02-detection-result\.png|03-android-flow\.png' README.md 2>$null).Count -gt 0){throw 'old media placeholders remain'}
if((rg -n 'docs/screenshots/01-main-window\.png|docs/screenshots/02-detection-result\.jpg|docs/screenshots/03-android-flow\.jpg|docs/media/demo-v1\.0\.5\.mp4' README.md).Count -lt 4){throw 'media links missing'}
~~~

Expected: no formatting errors, no old placeholder references, and all four new media paths appear.

- [ ] **Step 3: Re-run public link and secret checks**

Check the website, privacy page, terms page, GitHub repository, and v1.0.5 Release with HTTP requests. Scan tracked files for keystores, PFX/PEM keys, the Android signing password, server credentials, and SSH material. Expected: links return 200, private-file count is zero, and known-secret matches are zero.

### Task 4: Commit and push the media update

**Files:**
- Commit: README files and the eight mirrored media files.

- [ ] **Step 1: Commit and push the public repository**

Run:

~~~powershell
git add README.md docs/screenshots docs/media
git diff --cached --check
git commit -m "docs: add README showcase media"
git push origin main
~~~

Expected: `main` advances on `hiddenmoon-dusk/water-reaction-detection` and the four media files are available in the public tree.

- [ ] **Step 2: Commit the source mirror**

Run:

~~~powershell
git add README.md docs/screenshots docs/media
git diff --cached --check
git commit -m "docs: mirror README showcase media"
~~~

Expected: only the README and four media files are committed; existing `.superpowers/brainstorm/` user-owned directories remain untouched.

- [ ] **Step 3: Verify the public raw README and media paths**

Fetch the raw public README and confirm it contains the `产品展示` section and the four new relative paths. Confirm the public GitHub tree exposes each screenshot and MP4 without exposing the original local `展示素材` folder.
