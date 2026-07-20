# 水体反应管检测系统

水体反应管检测系统用于对水样照片中的反应管进行定位、分类、结果归档和联网同步，包含 Windows 桌面端、Android 客户端、模型工具和发布服务器。

## 功能概览

- 支持三类水样：污水、生活用水、养殖水体。
- Windows 桌面端支持默认检测、精细扫描、手动框选；结果保存到本地，并可在网络恢复后自动上传。
- Android 客户端支持 Android 14（API 34）及以上，模型离线内置，支持拍照、相册、检测、结果保存和联网同步。
- 服务器提供客户端注册、结果接收、统计展示、结果导出和桌面端/Android 发布管理。
- `models/` 中保留训练和桌面端使用的模型；Android APK 中包含转换后的 TFLite 模型。

## 正式发布准备

正式发布流程见 [`docs/正式发布指南.md`](docs/正式发布指南.md)。Android Release
必须使用仓库外的正式 keystore；Windows 正式版应使用 Inno Setup 安装包和
Authenticode 签名；服务器生产模式必须配置 HTTPS 和独立密钥。当前 v1.0.4
APK 使用 Debug 证书，只能用于验证，不应直接上架。

## 获取可运行程序

请打开仓库的 GitHub Releases 页面下载对应资产：

- Windows 完整包：`water-detection-desktop-v1.0.4-win64.zip`
- Windows 主程序：`water-detection-desktop-v1.0.4.exe`（仅主程序文件，推荐下载完整包）
- Android 安装包：`water-reaction-android-v1.0.4.apk`
- 校验文件：`SHA256SUMS.txt`

Windows 完整包内包含 EXE、模型文件和 `release.json`，请保持目录结构，不要只把 EXE 从目录中单独取出运行。Android 安装包只适用于 Android 14/API 34 及以上设备。

## Windows 快速使用

1. 下载并解压 Windows 完整包。
2. 双击 `水体反应管检测系统.exe`。
3. 选择水样类型，再选择图片。
4. 根据图片情况选择“默认检测”“精细扫描”或“手动框选”。
5. 检查检测结果后保存；结果会按水样类型归档。

如果状态栏显示网络不可用，检测和本地保存仍可继续，上传任务会留在本地队列中，网络恢复后自动重试。不要删除程序目录中的模型文件和 `release.json`。

## Android 快速使用

1. 在 Android 14/API 34 或更高版本设备上安装 APK。
2. 首次使用时按系统提示授予相机和照片访问权限。
3. 先选择水样类型，再选择拍照或从相册选择图片。
4. 选择检测模式并查看结果，确认后保存。
5. 有网络时，已保存结果会自动同步；同步失败时应用会保留待上传队列。

APK 内置检测模型，普通检测不依赖网络。应用更新、服务器地址和发布代次由发布包配置管理，请优先使用 Releases 中的完整版本。

## 从源码运行 Windows 客户端

项目需要 Python 3.13（x64）和 Windows 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-client.txt
python reaction_app.py
```

Streamlit 备用界面入口为 `app.py`，需要额外安装 Streamlit 后运行：

```powershell
python -m pip install streamlit
streamlit run app.py
```

## Android 源码构建

Android 工程位于 `android-app/`，目标 SDK 为 36，最低 SDK 为 34，构建使用 Java 17。模型转换产物位于本地的 `model-contract/generated/current/`，该目录属于构建产物并被 Git 忽略；直接安装时请使用 Releases 中的 APK。

在已经准备好 Android SDK、Java 17 和模型转换产物的环境中，可执行：

```powershell
cd android-app
.\gradlew.bat --no-daemon testDebugUnitTest
.\gradlew.bat --no-daemon assembleDebug
```

正式构建前请复制 `android-app/release-signing.properties.example` 为仓库外的
`android-app/release-signing.properties`，在本机填写正式签名信息，再设置服务器
环境变量并执行：

```powershell
$env:WATER_PUBLIC_BASE_URL = 'https://your-domain.example'
$env:WATER_BOOTSTRAP_TOKEN = '在本机输入的发布 Token'
.\scripts\build-formal-release.ps1
```

缺少签名配置时 `assembleRelease`/`bundleRelease` 会失败，不会退回 Debug 签名。

Windows 安装包使用 `scripts/build-windows-formal-release.ps1` 和
`packaging/windows/water-detection.iss` 生成；面向公众发布时还需要代码签名证书。

模型转换、样本一致性检查和 APK 生成流程请参考 `android-app/一键更新模型并生成APK.bat` 及 `model-tools/`。

## 服务器部署

服务器代码位于 `server/`，部署脚本位于 `server/deploy/`。首次部署前必须通过环境变量提供生产域名、证书邮箱、管理员初始密码和 Token，不能把密码写入脚本或提交到 Git：

```bash
export WATER_PUBLIC_DOMAIN='your-domain.example'
export WATER_CERTBOT_EMAIL='release-owner@example.com'
export WATER_PRODUCTION=true
export WATER_ADMIN_INITIAL_PASSWORD='请在本机输入长度至少 12 位的强密码'
export WATER_BOOTSTRAP_TOKEN='请在本机输入与发布包一致的 Token'
sudo -E bash server/deploy/bootstrap.sh /path/to/project/server
```

部署脚本会自动生成 Secret Key，强制 HTTPS，并在证书申请失败时停止部署。正式 Android 发布后，还应把 APK 证书 SHA-256 配置到 `WATER_ANDROID_SIGNING_CERT_SHA256`。数据库、检测结果、上传文件、签名私钥和密码文件均不属于 Git 仓库内容。

## 目录说明

```text
reaction_app.py       Windows PyQt5 桌面客户端
app.py                Streamlit 备用界面
android-app/          Android 客户端源码
models/               检测模型和分类模型
model-tools/          模型转换与一致性校验工具
server/               Flask 发布与结果同步服务
packaging/windows/    Inno Setup 安装包模板
scripts/              训练、标注、构建和端到端检查脚本
tests/                Windows 客户端测试
server/tests/         服务器测试
docs/中文使用说明.md  面向使用者的中文说明
docs/正式发布指南.md  正式构建、签名和服务器上线说明
```

## 安全与发布说明

- 不要提交 `.env`、密钥库、签名私钥、密码、数据库、样本、结果目录或 SDK 缓存。
- EXE、APK 和大型 ZIP 通过 GitHub Release 资产分发，不直接塞入 Git 提交历史。
- 下载后请使用 `SHA256SUMS.txt` 校验文件完整性。
- 本项目默认使用私有仓库；如果改为公开仓库，请先重新检查服务器域名、发布配置和所有部署资料。
