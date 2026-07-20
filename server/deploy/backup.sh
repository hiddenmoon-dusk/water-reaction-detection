#!/usr/bin/env bash
set -euo pipefail

DB="/opt/water-detection/instance/app.db"
BACKUPS="/opt/water-detection/storage/backups"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

mkdir -p "$BACKUPS"
sqlite3 "$DB" ".backup '$BACKUPS/app-$STAMP.db'"
find "$BACKUPS" -type f -name 'app-*.db' -mtime +14 -delete
