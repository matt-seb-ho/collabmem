import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int


def run_snippet(code: str, timeout: float = 2.0) -> RunResult:
    """Run arbitrary Python code in an isolated subprocess and capture stdout/stderr.

    `code` should be a complete Python script. Whatever it prints to stdout/stderr
    will be returned. You are responsible for having it call solve(), etc.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "user_code.py")

        # Dedent so you can pass triple-quoted strings nicely
        script = textwrap.dedent(code)

        with open(script_path, "w") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            # You can wrap this however you want
            raise RuntimeError(f"User code timed out after {timeout}s") from e

        return RunResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
