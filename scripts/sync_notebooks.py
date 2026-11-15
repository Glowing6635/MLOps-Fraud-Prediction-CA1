"""Sync saved notebook pairs and stage their Python representations before commit."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-optional-locks", *args], check=check, capture_output=True)


def message(status: str, text: str) -> None:
    print(f"[notebook-sync] {status} {text}", file=sys.stderr)


def cell_inputs(notebook) -> list[tuple[str, str]]:
    result = [
        (cell.cell_type, cell.source.replace("\r\n", "\n").rstrip("\n"))
        for cell in notebook.cells
    ]
    while result and result[-1][1] == "":
        result.pop()
    return result


@dataclass
class Update:
    script: Path
    notebook: Path
    direction: str
    script_before: bytes
    notebook_before: bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report required syncs without writing or staging files")
    args = parser.parse_args()
    os.chdir(os.fsdecode(git("rev-parse", "--show-toplevel").stdout).rstrip("\n"))
    paths = git("ls-files", "--cached", "-z", "--", "notebooks/").stdout.split(b"\0")
    pairs = []
    for raw in sorted(set(paths)):
        if raw:
            path = Path(os.fsdecode(raw))
            if path.suffix == ".py" and path.with_suffix(".ipynb").is_file():
                pairs.append((path, path.with_suffix(".ipynb")))
    if not pairs:
        message("OK", "No indexed notebook pairs to sync.")
        return 0

    try:
        import jupytext
        import nbformat
    except ImportError:
        message("ERROR", "Notebook dependencies missing. Run: uv sync --locked")
        return 1

    updates = []
    errors = False
    message("START", f"Checking {len(pairs)} saved notebook pair(s).")
    for script, notebook in pairs:
        try:
            if any(p.is_symlink() for p in (script, notebook, *script.parents)):
                raise ValueError("symlink pairs are unsupported; use regular files")
            if not script.is_file():
                raise ValueError("Python file is missing; restore it or stage its deletion")
            if git("diff", "--quiet", "--", script.as_posix(), check=False).returncode:
                raise ValueError("Python file has unstaged edits; stage the whole file or stash those edits first")
            staged = git("show", f":{script.as_posix()}").stdout.decode("utf-8")
            script_before = script.read_bytes()
            notebook_before = notebook.read_bytes()
            script_inputs = cell_inputs(jupytext.reads(staged, fmt="py:percent"))
            local = nbformat.reads(notebook_before.decode("utf-8"), as_version=4)
            nbformat.validate(local)
            notebook_inputs = cell_inputs(local)
            if script_inputs == notebook_inputs:
                continue

            previous = git("show", f"HEAD:{script.as_posix()}", check=False)
            if previous.returncode:
                raise ValueError("new pair differs; sync it manually, review the Python diff, and stage it first")
            base = cell_inputs(jupytext.reads(previous.stdout.decode("utf-8"), fmt="py:percent"))
            if script_inputs == base:
                if notebook.stat().st_mtime_ns < script.stat().st_mtime_ns:
                    raise ValueError("notebook is older than its Python file; reconcile the pair manually before committing")
                direction = "notebook-to-python"
            elif notebook_inputs == base:
                direction = "python-to-notebook"
            else:
                raise ValueError("both copies changed since HEAD; reconcile them manually, then stage the Python file")
            updates.append(Update(script, notebook, direction, script_before, notebook_before))
        except nbformat.ValidationError as error:
            message("ERROR", f"{script.as_posix()}: invalid notebook: {error.message}")
            errors = True
        except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as error:
            message("ERROR", f"{script.as_posix()}: {error}")
            errors = True

    if errors:
        message("STOP", "Commit blocked. No notebook files changed or staged.")
        return 1
    if args.check:
        for update in updates:
            message("PENDING", f"{update.script.as_posix()}: {update.direction}")
        message("STOP" if updates else "OK", "Sync required." if updates else "Notebook inputs match the index.")
        return int(bool(updates))

    # Preflight every pair before changing any of them. Editors should remain idle.
    for update in updates:
        if (update.script.read_bytes() != update.script_before
                or update.notebook.read_bytes() != update.notebook_before):
            message("STOP", f"{update.script}: files changed during the check; save and retry. Nothing synced or staged.")
            return 1

    had_staged_changes = bool(git("diff", "--cached", "--quiet", check=False).returncode)
    to_stage = []
    try:
        for update in updates:
            if update.direction == "notebook-to-python":
                arguments = ["--to", "py:percent", "--output", str(update.script), str(update.notebook)]
                to_stage.append(update.script.as_posix())
            else:
                arguments = ["--to", "ipynb", "--update", str(update.script)]
            result = subprocess.run(
                [sys.executable, "-m", "jupytext", *arguments], capture_output=True, text=True
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            converted = jupytext.reads(update.script.read_text(encoding="utf-8"), fmt="py:percent")
            saved = nbformat.reads(update.notebook.read_text(encoding="utf-8"), as_version=4)
            if cell_inputs(converted) != cell_inputs(saved):
                raise RuntimeError(f"{update.script}: conversion did not preserve cell inputs; review the pair manually")
            message("SYNC", f"{update.script.as_posix()}: {update.direction}")
        if to_stage:
            git("add", "--", *to_stage)
            message("STAGED", ", ".join(to_stage))
            if not had_staged_changes:
                message("NOTE", "Notebook edits were the only changes. If Git reports 'nothing to commit', retry the same commit command.")
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        message("ERROR", str(error))
        message("STOP", "Commit blocked. Inspect git diff and git diff --cached before retrying; some pairs may have synced.")
        return 1
    message("OK", f"{len(pairs)} pair(s) ready; {len(to_stage)} Python file(s) staged for this commit.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        message("ERROR", str(error))
        raise SystemExit(1)
