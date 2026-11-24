#!/usr/bin/env bash
# Updates an installed kfcollect from this folder. Config and data are kept.
#   sudo ./update.sh              update, restart, re-derive (all days if the derivation changed), post a report
#   sudo ./update.sh --no-report  skip the fresh report
#   sudo ./update.sh --rollback   go back to the version that was installed before
set -euo pipefail
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP=/opt/kfcollect
ETC=/etc/kfcollect
RUN=(sudo -u kfcollect env KF_CONFIG="$ETC/config.env" PYTHONPATH="$APP" "$APP/venv/bin/python" -m kfcollect)
version() { sed -n 's/^__version__ = "\(.*\)"/\1/p' "$1/__init__.py" 2>/dev/null || echo "?"; }
note() { (cd "$APP" && "${RUN[@]}" status --note "$1") >/dev/null 2>&1 || true; }

REPORT=1
case "${1:-}" in
  --no-report) REPORT=0 ;;
  --rollback)
    [[ -d "$APP/kfcollect.prev" ]] || { echo "No previous version to roll back to."; exit 1; }
    cur=$(version "$APP/kfcollect"); prev=$(version "$APP/kfcollect.prev")
    rm -rf "$APP/kfcollect.failed"; mv "$APP/kfcollect" "$APP/kfcollect.failed"; mv "$APP/kfcollect.prev" "$APP/kfcollect"
    systemctl restart kfcollect
    echo "Rolled back $cur -> $prev"; note "Rolled back $cur → $prev"
    exit 0 ;;
  "") ;;
  *) echo "unknown option: $1"; exit 2 ;;
esac

if [[ ! -d "$APP/kfcollect" ]]; then
  echo "kfcollect isn't installed yet; running install.sh instead."
  exec "$HERE/install.sh"
fi

OLD=$(version "$APP/kfcollect"); NEW=$(version "$HERE/kfcollect")
echo "==> Updating kfcollect $OLD -> $NEW"

# Did the derivation logic change? Then every day must be re-derived, not just the last few.
DERIVE_CHANGED=0
cmp -s "$APP/kfcollect/derive.py" "$HERE/kfcollect/derive.py" || DERIVE_CHANGED=1

# Derivations still running (an earlier update's background re-derive, the hourly timer, the daily report)
# have the old code loaded and would keep writing with it. Stop them; the re-derive below redoes their work,
# in full if an earlier update's re-derive was cut short.
running=$(systemctl list-units --type=service --state=active,activating --no-legend --plain \
  'kfcollect-rederive-*' kfcollect-derive.service kfcollect-report.service | awk '{print $1}')
if [[ -n "$running" ]]; then
  echo "==> Stopping derivations still running on $OLD: $(echo $running)"
  systemctl stop $running
  [[ "$running" == *kfcollect-rederive-* ]] && DERIVE_CHANGED=1
fi

echo "==> Code (previous version kept in $APP/kfcollect.prev)"
rm -rf "$APP/kfcollect.prev"
cp -a "$APP/kfcollect" "$APP/kfcollect.prev"
rm -rf "$APP/kfcollect"
cp -r "$HERE/kfcollect" "$APP/kfcollect"
find "$APP/kfcollect" -name '__pycache__' -prune -exec rm -rf {} +
chmod -R a+rX "$APP"

echo "==> Config: adding any new settings (existing values untouched)"
added=0
while IFS= read -r line; do
  [[ "$line" =~ ^(KF_[A-Z_]+)= ]] || continue
  key="${BASH_REMATCH[1]}"
  if ! grep -q "^$key=" "$ETC/config.env"; then
    (( added == 0 )) && printf '\n# Added by update to %s (defaults; edit as needed)\n' "$NEW" >> "$ETC/config.env"
    echo "$line" >> "$ETC/config.env"; echo "    + $key"; added=$((added + 1))
  fi
done < "$HERE/config.env.example"
(( added )) || echo "    nothing new"

echo "==> systemd units"
cp "$HERE"/systemd/*.service "$HERE"/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload
for t in status derive report; do systemctl enable --now "kfcollect-$t.timer" >/dev/null; done

echo "==> Restarting the collector"
systemctl restart kfcollect
sleep 15
if ! systemctl is-active --quiet kfcollect; then
  echo "!! The collector didn't stay up on $NEW. Rolling back to $OLD."
  journalctl -u kfcollect -n 30 --no-pager || true
  rm -rf "$APP/kfcollect.failed"; mv "$APP/kfcollect" "$APP/kfcollect.failed"; mv "$APP/kfcollect.prev" "$APP/kfcollect"
  systemctl restart kfcollect
  note "⚠️ Update to $NEW failed to start; rolled back to $OLD. Logs: journalctl -u kfcollect"
  exit 1
fi
age=$(python3 -c "import json,time;print(int(time.time()-json.load(open('/var/lib/kfcollect/heartbeat.json'))['ts']/1000))" 2>/dev/null || echo "?")
echo "    running (heartbeat ${age}s ago)"

echo "==> Self test"
(cd "$APP" && "${RUN[@]}" selftest) || echo "    (self test reported problems above; the collector is running)"

note "Updated $OLD → $NEW. Collector restarted and healthy."
if (( REPORT )); then
  if (( DERIVE_CHANGED )); then
    ARGS="derive --full"; WHAT="every day (the derivation changed)"
  else
    ARGS="derive --days 3"; WHAT="the last 3 days"
  fi
  UNIT="kfcollect-rederive-$(date +%Y%m%d-%H%M%S)"
  echo "==> Re-deriving $WHAT in the background ($UNIT), then posting a fresh report to Discord"
  # A transient systemd job keeps going if this SSH session ends.
  systemd-run --quiet --unit "$UNIT" --uid=kfcollect --gid=kfcollect --nice=10 \
    --working-directory="$APP" --setenv=KF_CONFIG="$ETC/config.env" --setenv=PYTHONPATH="$APP" \
    /bin/sh -c "$APP/venv/bin/python -m kfcollect $ARGS && $APP/venv/bin/python -m kfcollect report" \
    && echo "    follow it: journalctl -u $UNIT -f" \
    || echo "    couldn't start it; run by hand: kfcollect $ARGS && kfcollect report"
fi
echo
echo "Done: $OLD -> $NEW.  Roll back with: sudo $HERE/update.sh --rollback"
