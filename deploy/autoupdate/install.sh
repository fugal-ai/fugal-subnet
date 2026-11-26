#!/usr/bin/env bash
# Install the opt-in validator auto-updater. Run once, as root, from the
# checkout your validator runs (docs/VALIDATOR_GUIDE.md, "Auto-update (optional)"):
#
#   sudo deploy/autoupdate/install.sh
#
# Re-running it updates the updater itself and re-pins the allowed signers from
# this checkout; it keeps /etc/fugal/autoupdate.conf and the updater's state.
# Options: --baseline X.Y.Z (only on the first install: install releases newer
# than X.Y.Z; default the version in this checkout's pyproject.toml),
# --no-enable (install, do not start the timer).
set -euo pipefail

BASELINE=""
ENABLE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --baseline) BASELINE="$2"; shift 2 ;;
    --no-enable) ENABLE=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LIB=/usr/local/lib/fugal-autoupdate
CONF=/etc/fugal/autoupdate.conf
SIGNERS_DIR=/etc/fugal/autoupdate

for tool in git ssh-keygen python3 systemctl systemd-run journalctl; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "python3 must be 3.10 or newer" >&2; exit 1; }

# The program, root-owned: the timer never runs code the validator user can write.
install -d -m 0755 -o root -g root "$LIB"
install -m 0755 -o root -g root "$HERE/fugal_autoupdate.py" "$LIB/fugal_autoupdate.py"

# The trust root, pinned now from this checkout.
install -d -m 0755 -o root -g root "$SIGNERS_DIR"
install -m 0644 -o root -g root "$REPO/deploy/keys/allowed_signers" "$SIGNERS_DIR/allowed_signers"
echo "Pinned allowed signers (compare with deploy/keys/README.md):"
while read -r principal _opts keytype key _rest; do
  [ -n "${principal:-}" ] || continue
  printf '  %s  ' "$principal"
  printf '%s %s\n' "$keytype" "$key" | ssh-keygen -lf - | awk '{print $1, $2}'
done < "$SIGNERS_DIR/allowed_signers"

if [ ! -e "$CONF" ]; then
  install -d -m 0755 /etc/fugal
  install -m 0644 -o root -g root "$HERE/autoupdate.conf.example" "$CONF"
  user="$(sed -n 's/^service_user *= *//p' "$CONF")"
  uv_path=""
  if id "$user" >/dev/null 2>&1; then
    home="$(getent passwd "$user" | cut -d: -f6)"
    for c in /usr/local/bin/uv /usr/bin/uv "$home/.local/bin/uv" "$home/.cargo/bin/uv"; do
      if [ -x "$c" ]; then uv_path="$c"; break; fi
    done
  fi
  if [ -n "$uv_path" ]; then
    sed -i "s|^uv = .*|uv = $uv_path|" "$CONF"
    echo "uv: $uv_path"
  else
    echo "uv not found for user '$user': set 'uv =' in $CONF" >&2
  fi
  echo "Wrote $CONF: check base_dir, env_file, validator_args against your unit."
fi

install_root="$(sed -n 's/^install_root *= *//p' "$CONF")"
install_root="${install_root:-/opt/fugal-releases}"
install -d -m 0755 -o root -g root "$install_root"
dropin_dir="$(sed -n 's/^dropin_dir *= *//p' "$CONF")"
install -d -m 0755 -o root -g root "${dropin_dir:-/etc/systemd/system/fugal-validator.service.d}"

install -m 0644 -o root -g root "$HERE/fugal-autoupdate.service" /etc/systemd/system/
install -m 0644 -o root -g root "$HERE/fugal-autoupdate.timer" /etc/systemd/system/

if [ -n "$BASELINE" ]; then
  python3 "$LIB/fugal_autoupdate.py" init --baseline "$BASELINE"
else
  python3 "$LIB/fugal_autoupdate.py" init
fi

systemctl daemon-reload
if [ "$ENABLE" -eq 1 ]; then
  systemctl enable --now fugal-autoupdate.timer
  echo "Enabled. Status: sudo python3 $LIB/fugal_autoupdate.py status;" \
       "log: journalctl -u fugal-autoupdate"
else
  echo "Installed, timer not enabled: sudo systemctl enable --now fugal-autoupdate.timer"
fi
