# README 展示素材接入设计规格

## 目标

把用户提供的 Windows、检测结果、Android 流程截图和演示视频接入公开 README，替换原来的“素材待补充”说明，同时保证源码工作区和公开仓库使用相同的媒体路径。

## 素材映射

素材从 `F:/code/展示素材/` 复制到仓库的稳定 ASCII 路径：

| 原素材 | 仓库路径 | README 用途 |
| --- | --- | --- |
| `Windows 主界面截图.png` | `docs/screenshots/01-main-window.png` | Windows 主界面展示 |
| `检测结果截图.jpg` | `docs/screenshots/02-detection-result.jpg` | 检测框、标签、置信度展示 |
| `Android 流程截图.jpg` | `docs/screenshots/03-android-flow.jpg` | Android 选择照片到结果的流程展示 |
| `演示视频.mp4` | `docs/media/demo-v1.0.5.mp4` | 可点击的完整操作演示 |

使用 ASCII 文件名是为了降低 GitHub Markdown 对中文路径和 URL 编码的兼容风险。原始素材文件夹不修改。

## README 展示方式

1. 将三张截图放在“产品展示”区域，用 HTML `img` 控制宽度，避免原始大图撑破页面。
2. 用检测结果截图作为视频链接的可视封面：点击封面打开 `docs/media/demo-v1.0.5.mp4`。这样即使 GitHub 页面不自动播放仓库内 MP4，用户仍能直接访问视频文件。
3. 删除原来的媒体占位表和占位注释，改为真实素材展示；保留一句隐私提醒，说明截图和视频应使用演示数据。
4. 不对原始图片或视频做裁剪、压缩、加水印或内容修改。

## 同步范围

- 修改公开仓库 `README.md`。
- 将四个媒体文件复制到公开仓库对应路径。
- 将同样的 `README.md` 和媒体文件同步到 `F:/code/.worktrees/android-water-app/`。
- 不修改客户端、服务器、正式发布包和官网内容。

## 验收标准

- 四个目标媒体文件均存在，README 中所有媒体相对链接均指向实际文件。
- 三张图片在 GitHub Markdown 中使用相对路径，视频使用结果截图作为可点击封面。
- README 不再显示“展示素材待补充”或不存在的截图路径。
- MP4 可由 `ffprobe` 识别为 H.264、1280×720、约 43 秒视频。
- `git diff --check` 通过；公开仓库敏感文件扫描无命中。
- 公开 raw README、GitHub 页面和官网相关链接返回正常状态。

## 不在本次范围

- 不修改媒体内容和画面隐私。
- 不新增视频托管服务或 Git LFS 配置。
- 不把用户本地 `F:/code/展示素材/` 文件夹本身提交到仓库。
