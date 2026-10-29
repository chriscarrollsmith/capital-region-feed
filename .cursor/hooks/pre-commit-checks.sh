#!/usr/bin/env bash
# Gate agent `git commit` on the same ruff/ty checks as CI.
set -euo pipefail

input=$(cat)

# The hook process often has a shorter PATH than an interactive shell.
# Emit JSON without jq or python so a missing interpreter still fail-closes
# with a message instead of empty output.
emit_raw() {
  local permission=$1
  local msg=${2:-}
  if [[ -z "$msg" ]]; then
    printf '%s\n' "{\"permission\":\"${permission}\"}"
    return
  fi
  local escaped=${msg//\\/\\\\}
  escaped=${escaped//\"/\\\"}
  escaped=${escaped//$'\n'/\\n}
  escaped=${escaped//$'\r'/}
  printf '%s\n' "{\"permission\":\"${permission}\",\"user_message\":\"${escaped}\",\"agent_message\":\"${escaped}\"}"
}

trap 'status=$?; if [[ $status -ne 0 ]]; then emit_raw deny "pre-commit checks: hook failed before it could finish (status ${status})."; exit 0; fi' EXIT

resolve_python() {
  if command -v python >/dev/null 2>&1; then
    PYTHON=(python)
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
    return 0
  fi
  if [[ -n "${HOME:-}" && -x "${HOME}/.local/bin/uv" ]]; then
    PYTHON=("${HOME}/.local/bin/uv" run python)
    return 0
  fi
  return 1
}

if ! resolve_python; then
  emit_raw deny 'pre-commit checks: need python, python3, or uv on PATH to read the hook payload.'
  exit 0
fi

json_query() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r '.command // empty'
    return
  fi
  printf '%s' "$input" | "${PYTHON[@]}" -c 'import json,sys; print(json.load(sys.stdin).get("command") or "")'
}

emit() {
  local permission=$1
  local msg=${2:-}
  if command -v jq >/dev/null 2>&1; then
    if [[ -n "$msg" ]]; then
      jq -n --arg p "$permission" --arg m "$msg" '{permission:$p, user_message:$m, agent_message:$m}'
    else
      jq -n --arg p "$permission" '{permission:$p}'
    fi
    return
  fi
  PERMISSION="$permission" MESSAGE="$msg" "${PYTHON[@]}" -c 'import json,os; p=os.environ["PERMISSION"]; m=os.environ.get("MESSAGE") or ""; d={"permission":p}; print(json.dumps({**d, "user_message": m, "agent_message": m} if m else d))'
}

if ! command=$(json_query); then
  emit_raw deny 'pre-commit checks: could not parse the hook payload.'
  exit 0
fi

if [[ ! "$command" =~ git[[:space:]]+commit ]]; then
  emit allow
  exit 0
fi

deny() {
  emit deny "$1"
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
