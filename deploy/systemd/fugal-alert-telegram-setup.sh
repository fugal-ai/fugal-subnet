#!/bin/sh
# Point fugal-watch alerts at a Telegram bot. Run as root on the validator host.
#
# Before running: create a bot with @BotFather (/newbot) and copy its token;
# open a chat with the bot and send it any message (a bot cannot message a
# user who has never written to it). Then:
#
#   sudo deploy/systemd/fugal-alert-telegram-setup.sh
#
# The token is read without echo, never passed on a command line, and ends
# up only in /etc/fugal/watch.env (mode 0600). The chat id is read from the
# bot's own getUpdates feed, so there is nothing to look up by hand.
set -eu
env_file="${FUGAL_WATCH_ENV:-/etc/fugal/watch.env}"
[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 2; }
[ -f "$env_file" ] || { echo "$env_file not found; install deploy/env/watch.env.example first"; exit 2; }
printf 'Telegram bot token (input hidden): '
stty -echo; read -r token; stty echo; printf '\n'
[ -n "$token" ] || { echo "empty token"; exit 2; }
api="https://api.telegram.org/bot$token"
me="$(curl -fsS -m 20 "$api/getMe")" || { echo "getMe failed: token rejected or no network"; exit 1; }
echo "bot: $(printf '%s' "$me" | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["username"])')"
chat_id="$(curl -fsS -m 20 "$api/getUpdates" | python3 -c '
import json,sys
r=json.load(sys.stdin)["result"]
ids=[u.get("message",u.get("channel_post",{})).get("chat",{}).get("id") for u in r]
ids=[i for i in ids if i is not None]
print(ids[-1] if ids else "")')"
[ -n "$chat_id" ] || { echo "no chat found: send the bot one message in Telegram, then rerun"; exit 1; }
echo "chat id: $chat_id"
tmp="$(mktemp)"
grep -vE '^(FUGAL_ALERT_URL|FUGAL_ALERT_KIND|FUGAL_ALERT_CHAT_ID)=' "$env_file" > "$tmp"
printf 'FUGAL_ALERT_KIND=telegram\nFUGAL_ALERT_URL=%s/sendMessage\nFUGAL_ALERT_CHAT_ID=%s\n' "$api" "$chat_id" >> "$tmp"
install -m 0600 -o root -g root "$tmp" "$env_file"; rm -f "$tmp"
curl -fsS -m 20 -d "chat_id=$chat_id" --data-urlencode "text=fugal-watch on $(hostname): Telegram alerts configured $(date -u +%Y-%m-%dT%H:%MZ)" "$api/sendMessage" >/dev/null \
  && echo "test message sent; $env_file updated"
