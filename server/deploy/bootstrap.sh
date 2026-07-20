#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:?Usage: bootstrap.sh SOURCE_DIR}"
APP_ROOT="/opt/water-detection"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip nginx sqlite3 rsync certbot python3-certbot-nginx apksigner

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

if [ ! -f "$APP_ROOT/instance/secrets.env" ]; then
    : "${WATER_ADMIN_INITIAL_PASSWORD:?请先设置 WATER_ADMIN_INITIAL_PASSWORD，再执行部署}"
    SECRET_KEY="$(openssl rand -hex 32)"
    BOOTSTRAP_TOKEN="${WATER_BOOTSTRAP_TOKEN:-water-reaction-bootstrap-v1}"
    cat > "$APP_ROOT/instance/secrets.env" <<EOF
WATER_SECRET_KEY=$SECRET_KEY
WATER_ADMIN_INITIAL_PASSWORD=$WATER_ADMIN_INITIAL_PASSWORD
WATER_BOOTSTRAP_TOKEN=$BOOTSTRAP_TOKEN
WATER_APP_ROOT=$APP_ROOT
WATER_DATABASE=$APP_ROOT/instance/app.db
WATER_STORAGE_ROOT=$APP_ROOT/storage
EOF
    chmod 600 "$APP_ROOT/instance/secrets.env"
fi

install -m 0644 "$SOURCE_DIR/deploy/water-detection.service" /etc/systemd/system/water-detection.service
install -m 0644 "$SOURCE_DIR/deploy/nginx.conf" /etc/nginx/sites-available/water-detection
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

DOMAIN="hiddenmoon.duckdns.org"
CERTBOT_COMMON=(
    --nginx
    -d "$DOMAIN"
    --non-interactive
    --agree-tos
    --register-unsafely-without-email
    --redirect
)
if [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    if ! certbot "${CERTBOT_COMMON[@]}" --reinstall; then
        echo "WARNING: HTTPS certificate reinstall failed; HTTP remains available." >&2
    fi
elif ! certbot "${CERTBOT_COMMON[@]}"; then
    echo "WARNING: HTTPS certificate issuance failed; HTTP remains available." >&2
fi
