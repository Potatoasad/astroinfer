#!/usr/bin/env bash
# Run every tutorial notebook as a plain script; fail on the first error.
set -euo pipefail
cd "$(dirname "$0")/.."
for notebook in tutorials/[0-9]*.py; do
    echo "== $notebook"
    uv run python "$notebook" > /dev/null
done
echo "all tutorials ran"
