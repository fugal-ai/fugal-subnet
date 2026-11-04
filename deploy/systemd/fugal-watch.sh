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
#   FUGAL_ALERT_KIND         ntfy (plain text POST), discord (JSON webhook) or
#                            telegram (bot sendMessage; needs FUGAL_ALERT_CHAT_ID)
#   FUGAL_WATCH_SETTLE_BLOCKS  blocks into an epoch before it is judged (340).
#                            A validator sets weights after collection, at
#                            boundary + FUGAL_COLLECT_FRACTION x epoch (180 on a
#                            360-block epoch), and may use up to its 1800 s
#                            (150-block) post-collection budget verifying and
#                            scoring first, so a healthy validator can set
#                            weights as late as ~boundary + 330. Judging earlier
#                            reports "did not set weights this epoch" for one
#                            that simply hasn't got there yet. 340 is ~4 min
#                            before the next boundary. 0 = no wait.
#
# A run first waits until the current epoch is FUGAL_WATCH_SETTLE_BLOCKS old,
# then judges that epoch and exits (at most one epoch, ~72 min). systemd never
# starts a second instance while one is running, so an hourly timer against a
# 72-minute epoch simply skips a tick.
#
# The alert URL never goes on a command line: for Telegram it carries the bot
# token, and an ntfy topic or Discord webhook is a credential too. curl reads
# it from stdin (-K -), because argv is visible to every local user in ps.
set -u
cd "${FUGAL_WATCH_CHECKOUT:-/opt/fugal/fugal-subnet}" || exit 2
log="${FUGAL_WATCH_LOG:-/var/lib/fugal/watch-last.txt}"
tmp="$(mktemp)"
# Wait until the epoch has settled. Best effort: if the chain is unreachable
# the watch below reports that itself.
.venv/bin/python - "${FUGAL_WATCH_NETWORK:-test}" "${FUGAL_WATCH_EPOCH_INTERVAL:-4320}" \
  "${FUGAL_WATCH_SETTLE_BLOCKS:-340}" <<'PY' || true
import sys, time
import bittensor as bt
from fugal_subnet.benchmarks.slicer import blocks_per_epoch
network, interval, settle = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bpe = blocks_per_epoch(interval)
settle = max(0, min(settle, bpe - 1))
st = bt.Subtensor(network=network)
while settle:
    pos = st.get_current_block() % bpe
    if pos >= settle:
        break
    time.sleep(min(120, (settle - pos) * 12))
PY
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
  # The URL goes to curl on stdin as a config line, never on argv.
  post() { printf 'url = "%s"\n' "$FUGAL_ALERT_URL" | curl -fsS -m 20 -K - "$@" >/dev/null; }
  if [ -n "${FUGAL_ALERT_URL:-}" ]; then
    case "${FUGAL_ALERT_KIND:-ntfy}" in
      telegram)
        post -d "chat_id=${FUGAL_ALERT_CHAT_ID:-}" \
          --data-urlencode "text=$(printf '%s' "$body" | head -c 3900)" ;;
      discord)
        printf '%s' "$body" \
          | .venv/bin/python -c 'import json,sys;print(json.dumps({"content":sys.stdin.read()[:1900]}))' >"$tmp.json"
        post -H 'Content-Type: application/json' -d @"$tmp.json"
        rm -f "$tmp.json" ;;
      *)
        post -H "Title: fugal-watch FAIL" -H "Priority: high" -d "$body" ;;
    esac
  fi
  logger -t fugal-watch -p user.err "FAIL: exit $rc, $fails failed checks"
fi
rm -f "$tmp"
exit "$rc"
