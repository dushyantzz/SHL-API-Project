#!/usr/bin/env bash
# Point this repo at .githooks so Cursor co-author lines are removed on commit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$ROOT/.githooks/prepare-commit-msg"
git -C "$ROOT" config core.hooksPath .githooks
echo "Git hooks enabled: $ROOT/.githooks"
