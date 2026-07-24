#!/usr/bin/env bash
# Gate agent `git commit` on the same ruff/ty checks as CI.
set -euo pipefail

input=$(cat)
command=$(printf '%s' "$input" | jq -r '.command // empty')

if [[ ! "$command" =~ git[[:space:]]+commit ]]; then
  printf '%s\n' '{"permission":"allow"}'
  exit 0
fi

deny() {
  local msg=$1
  jq -n --arg m "$msg" '{
    permission: "deny",
    user_message: $m,
    agent_message: $m
  }'
  exit 0
}

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  UV=("${HOME}/.local/bin/uv")
else
  deny 'pre-commit checks: uv not found on PATH (required for ruff/ty).'
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

run_check() {
  local label=$1
  shift
  if ! env NO_COLOR=1 "$@" >"$tmp" 2>&1; then
    local body
    body=$(tail -c 4000 "$tmp")
    deny "pre-commit checks failed (${label}):
${body}"
  fi
}

run_check 'ruff check' "${UV[@]}" run ruff check .
run_check 'ruff format' "${UV[@]}" run ruff format --check .
run_check 'ty check' "${UV[@]}" run ty check

printf '%s\n' '{"permission":"allow"}'
exit 0
