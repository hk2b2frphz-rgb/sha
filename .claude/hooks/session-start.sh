#!/bin/bash
set -euo pipefail

# Only needed in Claude Code on the web / cloud sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

pip install --quiet --user vastai

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
fi

# VAST_API_KEY is set as an environment variable on the cloud environment
# (claude.ai/code -> environment settings), not committed to the repo.
if [ -n "${VAST_API_KEY:-}" ]; then
  "$HOME/.local/bin/vastai" set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
fi

# Codex runs on the ChatGPT subscription, not an API key. Its credentials live
# in ~/.codex/auth.json, which does not survive the container, so install the
# CLI here and log in non-interactively when a token is provided. Without one,
# run `codex login --device-auth` in the session: the browser flow cannot work
# headless but the device-code flow can.
if ! command -v codex >/dev/null 2>&1; then
  npm install -g @openai/codex >/dev/null 2>&1 || true
fi
if [ -n "${CODEX_ACCESS_TOKEN:-}" ] && command -v codex >/dev/null 2>&1; then
  printf '%s' "$CODEX_ACCESS_TOKEN" | codex login --with-access-token >/dev/null 2>&1 || true
fi
