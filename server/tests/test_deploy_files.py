from pathlib import Path


DEPLOY = Path("server/deploy")


def test_systemd_uses_one_worker_and_restarts():
    text = (DEPLOY / "water-detection.service").read_text(encoding="utf-8")

    assert "--workers 1" in text
    assert "--timeout 1800" in text
    assert "Restart=always" in text
    assert "EnvironmentFile=/opt/water-detection/instance/secrets.env" in text


def test_nginx_limits_uploads_and_proxies():
    text = (DEPLOY / "nginx.conf").read_text(encoding="utf-8")

    assert "client_max_body_size 64m;" in text
    assert "location = /admin/releases/desktop" in text
    assert "location = /admin/releases/bundle" in text
    bundle = text.split("location = /admin/releases/bundle {", 1)[1].split(
        "\n    }", 1
    )[0]
    assert "client_max_body_size 2g;" in bundle
    assert "client_body_timeout 1800s;" in bundle
    assert "proxy_read_timeout 1800s;" in bundle
    assert "proxy_send_timeout 1800s;" in bundle
    assert "location = /_desktop_release/desktop-latest.zip" in text
    assert "internal;" in text
    assert "alias /opt/water-detection/storage/releases/desktop-latest.zip;" in text
    assert 'location ~ "^/_mobile_release/' in text
    assert "(?<release_id>[0-9a-f]{32}-android)" in text
    assert (
        "alias /opt/water-detection/storage/releases/$release_id.apk;" in text
    )
    assert "default_type application/vnd.android.package-archive;" in text
    assert "error_page 403 404 = @mobile_release_unavailable;" in text
    assert "location @mobile_release_unavailable" in text
    assert "return 503" in text
    assert '\"code\":\"mobile_release_unavailable\"' in text
    assert text.count("internal;") >= 2
    assert text.count("sendfile on;") >= 2
    assert "proxy_set_header Host $host;" in bundle
    assert "proxy_set_header X-Real-IP $remote_addr;" in bundle
    assert (
        "proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;"
        in bundle
    )
    assert "proxy_set_header X-Forwarded-Proto $scheme;" in bundle
    assert "proxy_pass http://127.0.0.1:8000;" in bundle
    assert "server_name hiddenmoon.duckdns.org;" in text


def test_backup_uses_sqlite_online_backup():
    text = (DEPLOY / "backup.sh").read_text(encoding="utf-8")

    assert ".backup" in text
    assert "find" in text
    assert "-mtime +14" in text


def test_bootstrap_reloads_nginx_after_installing_site():
    text = (DEPLOY / "bootstrap.sh").read_text(encoding="utf-8")

    assert "apksigner" in text
    assert "systemctl restart water-detection" in text
    assert "systemctl reload nginx" in text
    assert "certbot" in text
    assert "--nginx" in text
    assert "--reinstall" in text
