#!/bin/sh
# Hourly liveness check for a Fugal validator host. Runs scripts/watch_subnet.py
# for one epoch boundary and raises an alert when any assertion fails or the
# run itself dies. Driven by fugal-watch.timer; settings in /etc/fugal/watch.env.
#
#   FUGAL_WATCH_NETWORK / FUGAL_WATCH_NETUID   which chain and subnet
#   FUGAL_WATCH_EPOCH_INTERVAL   tempo x 12 s (4320); the script defaults to 3600
#                            and would then watch the wrong boundaries
#   FUGAL_WATCH_VALIDATORS   name=hotkey_ss58,...  (this host's hotkey at least)
#   FUGAL_WATCH_MINERS       name=hotkey_ss58,...  (optional)
#   FUGAL_WATCH_REVEALS      name=/path/to/epochs,... (optional)
#   FUGAL_ALERT_URL          where failures go; empty means journal only
#   FUGAL_ALERT_KIND         ntfy (plain text POST) or discord (JSON webhook)
#
# One run lasts until the next epoch boundary (up to FUGAL_EPOCH_INTERVAL
# seconds). systemd never starts a second instance while one is running, so
# an hourly timer against a 72-minute epoch simply skips a tick.
set -u
cd "${FUGAL_WATCH_CHECKOUT:-/opt/fugal/fugal-subnet}" || exit 2
log="${FUGAL_WATCH_LOG:-/var/lib/fugal/watch-last.txt}"
tmp="$(mktemp)"
.venv/bin/python scripts/watch_subnet.py \
  --network "${FUGAL_WATCH_NETWORK:-test}" --netuid "${FUGAL_WATCH_NETUID:-552}" \
  --epochs 1 --timeout "${FUGAL_WATCH_TIMEOUT:-5400}" \
  --epoch-interval "${FUGAL_WATCH_EPOCH_INTERVAL:-4320}" \
  --validators "${FUGAL_WATCH_VALIDATORS:-}" --miners "${FUGAL_WATCH_MINERS:-}" \
  --reveals "${FUGAL_WATCH_REVEALS:-}" >"$tmp" 2>&1
rc=$?
cat "$tmp"
cp "$tmp" "$log"
if [ "$rc" -ne 0 ]; then
  fails="$(grep -c '^\s*\[FAIL\]' "$tmp")"
  body="fugal-watch on $(hostname): watch_subnet exit $rc, $fails FAIL lines
$(grep '\[FAIL\]' "$tmp" | head -10)"
  case "${FUGAL_ALERT_KIND:-ntfy}" in
    discord)
      [ -n "${FUGAL_ALERT_URL:-}" ] && printf '%s' "$body" \
        | .venv/bin/python -c 'import json,sys;print(json.dumps({"content":sys.stdin.read()[:1900]}))' \
        | curl -fsS -m 20 -H 'Content-Type: application/json' -d @- "$FUGAL_ALERT_URL" >/dev/null ;;
    *)
      [ -n "${FUGAL_ALERT_URL:-}" ] && curl -fsS -m 20 -H "Title: fugal-watch FAIL" -H "Priority: high" \
        -d "$body" "$FUGAL_ALERT_URL" >/dev/null ;;
  esac
  logger -t fugal-watch -p user.err "FAIL: exit $rc, $fails failed checks"
fi
rm -f "$tmp"
exit "$rc"
