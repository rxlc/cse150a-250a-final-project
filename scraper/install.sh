#!/usr/bin/env bash
# Installs (or upgrades) kfcollect on Ubuntu 22.04+.
#   sudo ./install.sh                                  # install / upgrade
#   sudo ./install.sh --webhook https://discord.com/api/webhooks/...   # also set the Discord webhook
# Re-running is safe: code is replaced, config and data are kept.
set -euo pipefail

WEBHOOK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --webhook) WEBHOOK="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo ./install.sh"; exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP=/opt/kfcollect
DATA=/var/lib/kfcollect
ETC=/etc/kfcollect

echo "==> Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv sqlite3 tzdata >/dev/null

echo "==> Service user and folders"
id kfcollect &>/dev/null || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin kfcollect
install -d -o kfcollect -g kfcollect -m 750 "$DATA"
install -d -o root -g kfcollect -m 750 "$ETC"
install -d -m 755 "$APP"

echo "==> Code"
rm -rf "$APP/kfcollect"
cp -r "$HERE/kfcollect" "$APP/kfcollect"
find "$APP/kfcollect" -name '__pycache__' -prune -exec rm -rf {} +
chmod -R a+rX "$APP"

echo "==> Python environment"
[[ -x "$APP/venv/bin/python" ]] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip

echo "==> Config"
if [[ ! -f "$ETC/config.env" ]]; then
  cp "$HERE/config.env.example" "$ETC/config.env"
  echo "    created $ETC/config.env"
else
  echo "    keeping existing $ETC/config.env"
fi
if [[ -n "$WEBHOOK" ]]; then
  if grep -q '^KF_DISCORD_WEBHOOK_URL=' "$ETC/config.env"; then
    sed -i "s|^KF_DISCORD_WEBHOOK_URL=.*|KF_DISCORD_WEBHOOK_URL=$WEBHOOK|" "$ETC/config.env"
  else
    echo "KF_DISCORD_WEBHOOK_URL=$WEBHOOK" >> "$ETC/config.env"
  fi
  echo "    Discord webhook set"
fi
chown root:kfcollect "$ETC/config.env"
chmod 640 "$ETC/config.env"

echo "==> Command wrapper: kfcollect"
cat > /usr/local/bin/kfcollect <<'EOF'
#!/usr/bin/env bash
# Runs kfcollect as the service user so it can read the data and config.
RUN=(env KF_CONFIG=/etc/kfcollect/config.env PYTHONPATH=/opt/kfcollect /opt/kfcollect/venv/bin/python -m kfcollect "$@")
cd /opt/kfcollect
if [[ "$(id -un)" == "kfcollect" ]]; then exec "${RUN[@]}"; fi
exec sudo -u kfcollect "${RUN[@]}"
EOF
chmod 755 /usr/local/bin/kfcollect

echo "==> systemd units"
cp "$HERE"/systemd/*.service "$HERE"/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now kfcollect.service
systemctl restart kfcollect.service
for t in status derive report; do systemctl enable --now "kfcollect-$t.timer"; done

echo "==> Clock (polls are timestamped; make sure NTP is on)"
timedatectl show -p NTPSynchronized --value 2>/dev/null | sed 's/^/    NTP synchronized: /' || true

echo "==> Self test"
cd "$APP"
if sudo -u kfcollect env KF_CONFIG="$ETC/config.env" "$APP/venv/bin/python" -m kfcollect selftest; then
  echo "    self test passed"
else
  echo "    self test reported problems (see above); the collector is still running"
fi

sleep 5
systemctl --no-pager --lines=5 status kfcollect.service || true
echo
echo "Done. Next:"
echo "  kfcollect status --test        # send a test message to Discord"
echo "  kfcollect status --print       # see collection status here"
echo "  journalctl -u kfcollect -f     # collector logs"
echo "  Config: $ETC/config.env   Data: $DATA"
