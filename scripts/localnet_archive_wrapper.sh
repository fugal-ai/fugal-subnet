#!/usr/bin/env bash
set -euo pipefail

# The pinned upstream localnet starts three authorities with the default state
# pruning policy. V2 acceptance verifies historical Commitment pallet values,
# so derive an ephemeral startup script that retains canonical block/state data.
# The image and its source script remain immutable; this wrapper is mounted
# read-only and the derived script exists only inside the disposable container.
source_script=/scripts/localnet.sh
archive_script=/scripts/fugal-localnet-archive.sh

awk '
  { print }
  /^[[:space:]]+--validator$/ {
    print "    --state-pruning=archive"
    print "    --blocks-pruning=archive"
  }
' "$source_script" > "$archive_script"
chmod 0700 "$archive_script"
exec "$archive_script" "$@"
