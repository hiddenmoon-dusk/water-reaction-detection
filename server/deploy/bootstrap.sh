#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: bootstrap.sh SOURCE_DIR}"
APP_ROOT="/opt/water-detection"
WATER_PUBLIC_DOMAIN="${WATER_PUBLIC_DOMAIN:?请设置 WATER_PUBLIC_DOMAIN，例如 detector.example.com}"
WATER_CERTBOT_EMAIL="${WATER_CERTBOT_EMAIL:?请设置 WATER_CERTBOT_EMAIL，用于 HTTPS 证书通知}"
WATER_ADMIN_INITIAL_PASSWORD="${WATER_ADMIN_INITIAL_PASSWORD:?请设置 WATER_ADMIN_INITIAL_PASSWORD}"
WATER_BOOTSTRAP_TOKEN="${WATER_BOOTSTRAP_TOKEN:?请设置 WATER_BOOTSTRAP_TOKEN}"

case "$WATER_PUBLIC_DOMAIN" in
    ""|.*|*.|*-|*/*|*\\*|*" "*)
        echo "WATER_PUBLIC_DOMAIN 不是安全的域名格式" >&2
        exit 1
        ;;
esac
if ! [[ "$WATER_PUBLIC_DOMAIN" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "WATER_PUBLIC_DOMAIN 只能包含字母、数字、点和连字符" >&2
    exit 1
fi
if [ "${#WATER_ADMIN_INITIAL_PASSWORD}" -lt 12 ]; then
    echo "WATER_ADMIN_INITIAL_PASSWORD 至少需要 12 个字符" >&2
    exit 1
fi
if [ "${#WATER_BOOTSTRAP_TOKEN}" -lt 16 ]; then
    echo "WATER_BOOTSTRAP_TOKEN 至少需要 16 个字符" >&2
    exit 1
fi
if [ ! -d "$SOURCE_DIR/water_server" ]; then
    echo "找不到服务器源码目录: $SOURCE_DIR/water_server" >&2
    exit 1
fi

WATER_PUBLIC_BASE_URL="https://${WATER_PUBLIC_DOMAIN}"
WATER_SECRET_KEY="${WATER_SECRET_KEY:-}"
if [ -z "$WATER_SECRET_KEY" ]; then
    WATER_SECRET_KEY="$(openssl rand -hex 32)"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y openssl python3-venv python3-pip nginx sqlite3 rsync certbot python3-certbot-nginx apksigner

if ! id waterapp >/dev/null 2>&1; then
    useradd --system --home "$APP_ROOT" --shell /usr/sbin/nologin waterapp
fi

if ! swapon --show | grep -q .; then
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

mkdir -p "$APP_ROOT"/{app,instance,storage/results,storage/releases,storage/temp,storage/backups,logs}
rsync -a --delete "$SOURCE_DIR/water_server/" "$APP_ROOT/app/water_server/"
install -m 0644 "$SOURCE_DIR/wsgi.py" "$APP_ROOT/app/wsgi.py"
install -m 0644 "$SOURCE_DIR/requirements.txt" "$APP_ROOT/app/requirements.txt"

python3 -m venv "$APP_ROOT/venv"
"$APP_ROOT/venv/bin/pip" install --upgrade pip
"$APP_ROOT/venv/bin/pip" install -r "$APP_ROOT/app/requirements.txt"

if [ -f "$APP_ROOT/instance/secrets.env" ]; then
    if ! grep -q '^WATER_PRODUCTION=true$' "$APP_ROOT/instance/secrets.env"; then
        echo "已有 secrets.env 不是生产配置，拒绝覆盖或继续部署" >&2
        exit 1
    fi
    if ! grep -qx "WATER_PUBLIC_DOMAIN=$WATER_PUBLIC_DOMAIN" "$APP_ROOT/instance/secrets.env" || \
        ! grep -qx "WATER_PUBLIC_BASE_URL=$WATER_PUBLIC_BASE_URL" "$APP_ROOT/instance/secrets.env"; then
        echo "已有 secrets.env 的生产域名与本次部署参数不一致，拒绝继续部署" >&2
        exit 1
    fi
else
    umask 077
    cat > "$APP_ROOT/instance/secrets.env" <<EOF
WATER_PRODUCTION=true
WATER_SECRET_KEY=$WATER_SECRET_KEY
WATER_ADMIN_INITIAL_PASSWORD=$WATER_ADMIN_INITIAL_PASSWORD
WATER_BOOTSTRAP_TOKEN=$WATER_BOOTSTRAP_TOKEN
WATER_PUBLIC_DOMAIN=$WATER_PUBLIC_DOMAIN
WATER_PUBLIC_BASE_URL=$WATER_PUBLIC_BASE_URL
WATER_CERTBOT_EMAIL=$WATER_CERTBOT_EMAIL
WATER_APP_ROOT=$APP_ROOT
WATER_DATABASE=$APP_ROOT/instance/app.db
WATER_STORAGE_ROOT=$APP_ROOT/storage
EOF
    chmod 600 "$APP_ROOT/instance/secrets.env"
fi

install -m 0644 "$SOURCE_DIR/deploy/water-detection.service" /etc/systemd/system/water-detection.service
NGINX_RENDERED="$(mktemp)"
trap 'rm -f "$NGINX_RENDERED"' EXIT
sed "s/__WATER_PUBLIC_DOMAIN__/$WATER_PUBLIC_DOMAIN/g" \
    "$SOURCE_DIR/deploy/nginx.conf" > "$NGINX_RENDERED"
install -m 0644 "$NGINX_RENDERED" /etc/nginx/sites-available/water-detection
ln -sfn /etc/nginx/sites-available/water-detection /etc/nginx/sites-enabled/water-detection
rm -f /etc/nginx/sites-enabled/default
install -m 0755 "$SOURCE_DIR/deploy/backup.sh" /usr/local/sbin/water-detection-backup
install -m 0644 "$SOURCE_DIR/deploy/logrotate.conf" /etc/logrotate.d/water-detection
cat > /etc/cron.d/water-detection-backup <<'EOF'
17 3 * * * waterapp /usr/local/sbin/water-detection-backup
EOF

chown -R waterapp:waterapp "$APP_ROOT"
systemctl daemon-reload
systemctl enable --now water-detection
systemctl restart water-detection
nginx -t
systemctl enable --now nginx
systemctl reload nginx

CERTBOT_COMMON=(
    --nginx
    -d "$WATER_PUBLIC_DOMAIN"
    --non-interactive
    --agree-tos
    --email "$WATER_CERTBOT_EMAIL"
    --redirect
)
if [ -d "/etc/letsencrypt/live/$WATER_PUBLIC_DOMAIN" ]; then
    certbot "${CERTBOT_COMMON[@]}" --reinstall
else
    certbot "${CERTBOT_COMMON[@]}"
fi
