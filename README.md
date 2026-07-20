# 水体反应管检测系统

本项目包含 Windows 桌面检测程序、Android 客户端、离线模型、模型转换工具和结果/发布服务器。

## 当前正式发布要求

- Android 14/API 34 及以上；目标 SDK/API 36。
- 正式 Android 包必须通过 `android-app/scripts/build-formal-release.ps1` 构建，并使用仓库外的正式 keystore；Debug APK 只能用于测试。
- Windows 正式版应使用 `scripts/build-windows-formal-release.ps1` 生成便携 ZIP 和 Inno Setup 安装包，并使用 Authenticode 证书签名。
- 服务器正式部署必须设置 `WATER_PRODUCTION=true`、HTTPS 域名、强管理员密码和独立 Token。
- 发布准备模板集中在 `F:\code\正式发布准备-v1.0.5`，其中不保存真实密码、私钥或服务器 secrets.env。

## 目录

```text
android-app/       Android 客户端和正式签名/构建脚本
model-tools/       模型转换与一致性检查工具
server/            Flask/Gunicorn 结果与发布服务器
packaging/windows/ Inno Setup 安装包模板
scripts/           Windows 正式发布和验证脚本
docs/              设计、运维和中文使用文档
assets/branding/   Android、Windows、官网共用的应用图标源和 PNG
```

## 应用图标

正式版统一使用“反应管信号”图标：青色底、白色反应管、荧光黄反应液和黑色检测刻线。
公开素材位于 `assets/branding/water-reaction-icon.png`，SVG 源位于
`assets/branding/water-reaction-icon.svg`；Windows 安装程序使用
`packaging/windows/water-reaction-icon.ico`，Android 使用自适应图标资源。

## 正式构建入口

Android：

```powershell
cd android-app
Copy-Item .\release-signing.properties.example .\release-signing.properties
# 在本机编辑 release-signing.properties，不要提交 Git
$env:WATER_PUBLIC_BASE_URL = 'https://your-domain.example'
$env:WATER_BOOTSTRAP_TOKEN = '在本机输入的发布 Token'
.\scripts\build-formal-release.ps1
```

Windows：

```powershell
$env:WATER_PUBLIC_BASE_URL = 'https://your-domain.example'
$env:WATER_BOOTSTRAP_TOKEN = '在本机输入的发布 Token'
..\scripts\build-windows-formal-release.ps1 -RequireInstaller
```

正式构建脚本会在缺少签名配置、模型资产、HTTPS 地址或必要工具时失败，不会把 Debug 包或未配置服务器的包标记为正式发布。

## 安全和数据

不要提交 keystore、PFX/PEM 私钥、密码、服务器 SSH 资料、数据库、上传结果、原始照片、缓存或构建产物。正式发布前还需完成隐私政策、用户协议、项目许可证和第三方模型/依赖授权核对。
