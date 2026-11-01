"""Start the pre-commit hook with a usable Bash.

On Windows, an unqualified ``bash`` is often the WSL relay, which cannot run
the hook script. Prefer Git Bash there. On Unix, run the shell gate with the
system Bash so Cloud Agent / Linux commits still get the ruff/ty checks.
"""

from __future__ import annotations

import os
import subprocess
import sys

_DENY = (
    '{"permission":"deny","user_message":'
    '"pre-commit checks: Bash was not found.",'
    '"agent_message":"pre-commit checks: Bash was not found."}\n'
)


def _git_bash() -> str | None:
    program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
    candidates = [
        os.path.join(program_files, 'Git', 'bin', 'bash.exe'),
        os.path.join(program_files, 'Git', 'usr', 'bin', 'bash.exe'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _bash() -> str | None:
    if os.name == 'nt':
        return _git_bash()
    for path in ('/bin/bash', '/usr/bin/bash'):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def main() -> int:
    bash = _bash()
    if bash is None:
        sys.stdout.write(_DENY)
        return 0
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre-commit-checks.sh')
    payload = sys.stdin.buffer.read()
    completed = subprocess.run([bash, script], input=payload, check=False)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
