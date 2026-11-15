"""Test automatic sync and staging in disposable repositories."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import jupytext
import nbformat

PROJECT = Path(__file__).resolve().parents[1]


class NotebookHookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="notebook-hook-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = os.environ.copy()
        for key in subprocess.check_output(["git", "rev-parse", "--local-env-vars"], text=True).splitlines():
            self.env.pop(key, None)
        self.env.update(UV_PROJECT_ENVIRONMENT=sys.prefix,
                        UV_CACHE_DIR=str(self.root / ".uv-cache"),
                        JUPYTER_DATA_DIR=str(self.root / ".jupyter"))
        self.git("init", "--quiet")
        self.git("config", "user.name", "Notebook test")
        self.git("config", "user.email", "notebook-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", ".githooks")
        shutil.copytree(PROJECT / ".githooks", self.root / ".githooks")
        (self.root / "scripts").mkdir()
        shutil.copy2(PROJECT / "scripts/sync_notebooks.py", self.root / "scripts/sync_notebooks.py")
        (self.root / "pyproject.toml").write_text('[project]\nname = "hook-test"\nversion = "0.0.0"\n')
        (self.root / "jupytext.toml").write_text('formats = "ipynb,py:percent"\n')
        (self.root / ".gitignore").write_text("*.ipynb\n")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, env=self.env, stderr=subprocess.PIPE)

    def pair(self, name="main", source="x = 1", committed=True):
        script = self.root / "notebooks" / f"{name}.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
        script.write_text(jupytext.writes(notebook, fmt="py:percent"), encoding="utf-8")
        self.git("add", "--", script.relative_to(self.root).as_posix())
        if committed:
            self.git("-c", "core.hooksPath=.no-hooks", "commit", "--quiet", "-m", "baseline")
        local = script.with_suffix(".ipynb")
        nbformat.write(notebook, local)
        return script, local

    @staticmethod
    def edit_notebook(local: Path, source: str) -> None:
        """Save new cell source and make the notebook newer than its script."""
        notebook = nbformat.read(local, as_version=4)
        notebook.cells[0].source = source
        nbformat.write(notebook, local)
        script = local.with_suffix(".py")
        newer: int = int(max(script.stat().st_mtime_ns, local.stat().st_mtime_ns)) + 2_000_000_000
        os.utime(local, ns=(newer, newer))

    def run_hook(self, *args):
        command = ([sys.executable, "-B", "scripts/sync_notebooks.py", *args] if args else
                   ["git", "hook", "run", "pre-commit"])
        return subprocess.run(command, cwd=self.root, env=self.env, text=True, capture_output=True)

    def snapshot(self):
        # Git itself may refresh index stat data before invoking a hook.
        # Compare staged object IDs/modes rather than that filesystem cache.
        return {
            "index": self.git("ls-files", "--stage", "-z"),
            "files": {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()
                      and p.suffix in {".py", ".ipynb"}},
        }

    def assert_unchanged(self, before):
        self.assertEqual(self.snapshot(), before)

    def test_no_pairs(self):
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[notebook-sync] OK", result.stderr)

    def test_matching_pair_is_unchanged(self):
        self.pair()
        before = self.snapshot()
        self.assertEqual(self.run_hook().returncode, 0)
        self.assert_unchanged(before)

    def test_notebook_edits_are_synced_and_staged(self):
        script, local = self.pair()
        self.edit_notebook(local, "x = 2")
        saved = local.read_bytes()
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("x = 2", script.read_text())
        self.assertIn(b"x = 2", self.git("show", ":notebooks/main.py"))
        self.assertEqual(local.read_bytes(), saved)
        self.assertIn("[notebook-sync] SYNC", result.stderr)
        self.assertIn("[notebook-sync] STAGED", result.stderr)

    def test_real_commit_contains_synced_notebook(self):
        _, local = self.pair()
        self.edit_notebook(local, "x = 99")
        (self.root / "note.md").write_text("Some other staged change\n")
        self.git("add", "note.md")
        self.git("commit", "--quiet", "-m", "exercise hook")
        self.assertIn(b"x = 99", self.git("show", "HEAD:notebooks/main.py"))
        self.assertNotIn(b".ipynb", self.git("ls-tree", "-r", "--name-only", "HEAD"))

    def test_notebook_only_commit_after_explicit_hook(self):
        _, local = self.pair()
        self.edit_notebook(local, "x = 42")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.git("commit", "--quiet", "-m", "notebook-only change")
        self.assertIn(b"x = 42", self.git("show", "HEAD:notebooks/main.py"))

    def test_notebook_only_commit_can_be_retried_if_git_rejects_initially_empty_index(self):
        _, local = self.pair()
        self.edit_notebook(local, "x = 42")
        result = subprocess.run(["git", "commit", "--quiet", "-m", "notebook-only change"],
                                cwd=self.root, env=self.env, capture_output=True, text=True)
        if result.returncode:
            self.assertIn("retry the same commit", result.stderr)
            self.assertIn(b"x = 42", self.git("show", ":notebooks/main.py"))
            self.git("commit", "--quiet", "-m", "notebook-only retry")
        self.assertIn(b"x = 42", self.git("show", "HEAD:notebooks/main.py"))

    def test_script_edit_updates_notebook_and_preserves_outputs(self):
        script, local = self.pair()
        notebook = nbformat.read(local, as_version=4)
        notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="old output\n")]
        notebook.cells[0].execution_count = 1
        nbformat.write(notebook, local)
        script.write_text(script.read_text().replace("x = 1", "x = 2"))
        self.git("add", "notebooks/main.py")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        notebook = nbformat.read(local, as_version=4)
        self.assertEqual(notebook.cells[0].source, "x = 2")
        self.assertEqual(notebook.cells[0].outputs[0].text, "old output\n")

    def test_unstaged_and_partially_staged_scripts_block_without_writes(self):
        script, local = self.pair()
        script.write_text(script.read_text().replace("x = 1", "x = 2"))
        self.edit_notebook(local, "x = 3")
        for partial in [False, True]:
            with self.subTest(partial=partial):
                if partial:
                    self.git("add", "notebooks/main.py")
                    script.write_text(script.read_text().replace("x = 2", "x = 4"))
                before = self.snapshot()
                result = self.run_hook()
                self.assertEqual(result.returncode, 1)
                self.assertIn("unstaged edits", result.stderr)
                self.assert_unchanged(before)

    def test_both_copies_changed_blocks_without_writes(self):
        script, local = self.pair()
        script.write_text(script.read_text().replace("x = 1", "x = 2"))
        self.git("add", "notebooks/main.py")
        self.edit_notebook(local, "x = 3")
        before = self.snapshot()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1)
        self.assertIn("both copies changed", result.stderr)
        self.assert_unchanged(before)

    def test_stale_notebook_blocks_without_writes(self):
        script, local = self.pair()
        self.edit_notebook(local, "old_branch = True")
        newer = local.stat().st_mtime_ns + 2_000_000_000
        os.utime(script, ns=(newer, newer))
        before = self.snapshot()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1)
        self.assertIn("notebook is older", result.stderr)
        self.assert_unchanged(before)

    def test_one_bad_pair_prevents_syncing_other_pairs(self):
        _, good = self.pair("good")
        _, bad = self.pair("bad")
        self.edit_notebook(good, "x = 2")
        bad.write_text("{invalid json")
        before = self.snapshot()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1)
        self.assertIn("[notebook-sync] ERROR", result.stderr)
        self.assert_unchanged(before)

    def test_new_matching_pair_is_accepted(self):
        self.pair(committed=False)
        self.assertEqual(self.run_hook().returncode, 0)

    def test_new_different_pair_requires_manual_sync(self):
        _, local = self.pair(committed=False)
        self.edit_notebook(local, "x = 2")
        before = self.snapshot()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1)
        self.assertIn("new pair differs", result.stderr)
        self.assert_unchanged(before)

    def test_missing_notebook_is_skipped(self):
        _, local = self.pair()
        local.unlink()
        self.assertEqual(self.run_hook().returncode, 0)

    def test_deleted_pair_is_not_resurrected(self):
        self.pair()
        self.git("rm", "--quiet", "notebooks/main.py")
        self.assertEqual(self.run_hook().returncode, 0)
        self.assertFalse((self.root / "notebooks/main.py").exists())

    def test_magic_cells_and_spaces_in_paths(self):
        script, local = self.pair("sub folder/my notebook")
        source = '%%load_clean default\nfrom mlops_fraud_prediction_ca1 import main'
        self.edit_notebook(local, source)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(jupytext.reads(script.read_text(), fmt="py:percent").cells[0].source, source)
        self.assertIn(b"load_clean", self.git("show", ":notebooks/sub folder/my notebook.py"))

    def test_metadata_and_empty_cells_do_not_trigger_writes(self):
        _, local = self.pair(source="x = 1\n\n")
        notebook = nbformat.read(local, as_version=4)
        notebook.metadata["custom"] = "local only"
        notebook.cells.append(nbformat.v4.new_markdown_cell(""))
        nbformat.write(notebook, local)
        before = self.snapshot()
        self.assertEqual(self.run_hook().returncode, 0)
        self.assert_unchanged(before)

    def test_check_mode_never_writes_or_stages(self):
        _, local = self.pair()
        self.edit_notebook(local, "x = 2")
        before = self.snapshot()
        result = self.run_hook("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[notebook-sync] PENDING", result.stderr)
        self.assert_unchanged(before)

    def test_magic_cells_roundtrip_without_execution_or_output_export(self):
        script, local = self.pair()
        sources = ["%load_ext load_clean", "%time result = 1 + 1", "!echo hello",
                   "%%load_clean default\nfrom mlops_fraud_prediction_ca1 import main",
                   "raise AssertionError('the hook must not execute cells')"]
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
        notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="LOCAL_OUTPUT_ONLY")]
        nbformat.write(notebook, local)
        newer = script.stat().st_mtime_ns + 2_000_000_000
        os.utime(local, ns=(newer, newer))
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c.source for c in jupytext.reads(script.read_text(), fmt="py:percent").cells], sources)
        self.assertNotIn(b"LOCAL_OUTPUT_ONLY", self.git("show", ":notebooks/main.py"))

    def test_crlf_working_script_and_utf8_notebook(self):
        self.git("config", "core.autocrlf", "true")
        script, local = self.pair()
        script.write_bytes(script.read_bytes().replace(b"\n", b"\r\n"))
        self.edit_notebook(local, 'label = "fraud \u2713"')
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('label = "fraud \u2713"', self.git("show", ":notebooks/main.py").decode("utf-8"))

    def test_missing_dependencies_have_actionable_message(self):
        self.pair()
        result = subprocess.run([sys.executable, "-S", "scripts/sync_notebooks.py"],
                                cwd=self.root, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("[notebook-sync] ERROR", result.stderr)
        self.assertIn("uv sync --locked", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_lossy_conversion_is_not_staged(self):
        _, local = self.pair()
        self.edit_notebook(local, "# %%\nx = 2")
        staged = self.git("show", ":notebooks/main.py")
        saved = local.read_bytes()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1)
        self.assertIn("conversion did not preserve cell inputs", result.stderr)
        self.assertEqual(self.git("show", ":notebooks/main.py"), staged)
        self.assertEqual(local.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
