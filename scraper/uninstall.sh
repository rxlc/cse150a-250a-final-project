#!/usr/bin/env bash
# Stops and removes kfcollect. Data in /var/lib/kfcollect is kept unless --purge is given.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo"; exit 1; }
for u in kfcollect.service kfcollect-status.timer kfcollect-derive.timer kfcollect-report.timer; do
  systemctl disable --now "$u" 2>/dev/null || true
done
rm -f /etc/systemd/system/kfcollect*.service /etc/systemd/system/kfcollect*.timer /usr/local/bin/kfcollect
systemctl daemon-reload
rm -rf /opt/kfcollect
if [[ "${1:-}" == "--purge" ]]; then
  rm -rf /var/lib/kfcollect /etc/kfcollect
  userdel kfcollect 2>/dev/null || true
  echo "Removed kfcollect, its data and config."
else
  echo "Removed kfcollect. Kept data (/var/lib/kfcollect) and config (/etc/kfcollect)."
fi
