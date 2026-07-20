# 水体检测平台运维说明

## 交付位置

- Windows 解压目录：`dist/水体反应管检测系统`
- 用户下载 ZIP：`output/water-detection-desktop.zip`
- 线上地址：由服务器的 `WATER_PUBLIC_BASE_URL` 配置，示例使用
  `https://your-domain.example`
- 服务器程序：`/opt/water-detection/app`
- 数据库：`/opt/water-detection/instance/app.db`
- 检测结果：`/opt/water-detection/storage/results`
- 当前桌面包：`/opt/water-detection/storage/releases/desktop-latest.zip`

桌面程序必须与 `reaction_classifier.h5`、`yolov8n.pt` 和
`release.json` 放在同一目录。两个模型没有嵌入 EXE。

## 日常管理

从主页右上角进入“管理员登录”。初始密码由部署环境变量
`WATER_ADMIN_INITIAL_PASSWORD` 设置；首次正式使用后应在管理页修改。

管理页支持：

1. 下载检测结果 ZIP。解压后包含“污水”“生活用水”“养殖水体”三个目录。
2. 上传新的电脑端 ZIP。ZIP 必须包含 EXE、`reaction_classifier.h5` 和
   `yolov8n.pt`。
3. 手机端上传按钮暂为占位功能。
4. 清除检测结果。该操作需要再次输入管理员密码。

## 发布顺序

清除检测结果会提升数据代次，使旧客户端不能继续上传旧队列。因此推荐顺序是：

1. 下载并留存旧检测结果。
2. 清除检测结果。
3. 上传新的电脑端 ZIP。
4. 从主页下载一次最新 ZIP，核对能够正常解压和启动。

如果先上传新版本、后清除结果，必须在清除后再次上传桌面 ZIP，否则刚发布的
`release.json` 会携带旧数据代次。

构建本地候选包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
```

服务器会在管理员上传时验证包内容，注入新的发布 ID、模型代次和数据代次，
并原子替换旧下载包。

## 服务管理

```bash
sudo systemctl status water-detection
sudo systemctl restart water-detection
sudo systemctl status nginx
sudo nginx -t
```

日志：

```bash
sudo tail -f /opt/water-detection/logs/error.log
sudo tail -f /opt/water-detection/logs/access.log
sudo journalctl -u water-detection -f
```

公开统计健康检查（将域名替换为生产值）：

```bash
curl -fsS https://your-domain.example/api/v1/public/statistics
```

正式首次部署必须先在服务器本机设置 `WATER_PUBLIC_DOMAIN`、
`WATER_CERTBOT_EMAIL`、`WATER_ADMIN_INITIAL_PASSWORD` 和
`WATER_BOOTSTRAP_TOKEN`，再执行：

```bash
sudo -E bash server/deploy/bootstrap.sh /path/to/project/server
```

脚本会拒绝缺少生产密钥、HTTP 公网地址、弱管理员密码和非安全域名，
不会在 HTTPS 申请失败时继续以 HTTP 作为正式服务。

证书自动续期检查：

```bash
sudo certbot renew --dry-run --no-random-sleep-on-renew
```

## 备份

`/etc/cron.d/water-detection-backup` 每天 03:17 运行在线 SQLite 备份，并保留
14 天。手动执行：

```bash
sudo -u waterapp /usr/local/sbin/water-detection-backup
ls -lh /opt/water-detection/storage/backups
```

恢复前先停止应用服务，并同时保存当前数据库和结果目录。

## 验证脚本

只验证客户端注册、样本上传、重复上传和公开统计：

```powershell
python scripts\e2e_smoke.py
```

同时验证管理员导出并清除测试数据：

```powershell
python scripts\e2e_smoke.py --admin-password "<管理员密码>" --clear
```

执行 `--clear` 后必须按“发布顺序”重新上传桌面 ZIP。

## 已知边界

- 已在当前 Windows 11 x64 机器完成 EXE 启动和模型加载验证。
- 尚未在独立、全新安装的 Windows 10/11 虚拟机上验证。
- 大陆不同运营商的长期可达性不能由单台开发机器保证；客户端已使用本地持久
  队列和自动重试，但仍应持续监控服务器访问日志和失败率。
