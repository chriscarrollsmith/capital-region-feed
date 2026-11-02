"""Start the pre-commit hook with Git Bash.

On Windows, an unqualified ``bash`` is the WSL relay, which cannot run the
hook script. This launcher picks Git Bash and forwards the hook payload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_DENY = (
    '{"permission":"deny","user_message":'
    '"pre-commit checks: Git Bash was not found.",'
    '"agent_message":"pre-commit checks: Git Bash was not found."}\n'
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
    """Prefer Git Bash. On non-Windows, PATH bash can run the hook."""
    git_bash = _git_bash()
    if git_bash is not None:
        return git_bash
    # An unqualified bash on Windows is often the WSL relay, which cannot
    # run this hook. Only fall back where bash is the system shell.
    if os.name == 'nt':
        return None
    return shutil.which('bash')


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
