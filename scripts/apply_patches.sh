#!/usr/bin/env bash
# Apply the documented patches (docs/PATCHES.md) to third_party/mados. Idempotent.
# Only needed for env-h100 (mmcv-lite); env-orig runs the submodule unpatched.
#   bash scripts/apply_patches.sh            apply
#   bash scripts/apply_patches.sh --revert   remove
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$REPO/third_party/mados"
for p in "$REPO"/scripts/patches/*.patch; do
    name="$(basename "$p")"
    if [ "${1:-}" = "--revert" ]; then
        if git -C "$SUB" apply --reverse --check "$p" 2>/dev/null; then
            git -C "$SUB" apply --reverse "$p"; echo "reverted $name"
        else
            echo "not applied: $name"
        fi
    elif git -C "$SUB" apply --reverse --check "$p" 2>/dev/null; then
        echo "already applied: $name"
    else
        git -C "$SUB" apply "$p"; echo "applied $name"
    fi
done
