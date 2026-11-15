"""Exercise NotebookHelpers magics in IPython without depending on an editor."""

from contextlib import redirect_stdout
import importlib
from io import StringIO
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config

from load_clean.linked import fingerprint


class NotebookMagicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = tempfile.TemporaryDirectory(prefix="notebook-magic-profile-")
        cls.addClassCleanup(cls.profile.cleanup)
        config = Config()
        config.HistoryManager.enabled = False
        with patch.dict(os.environ, {"IPYTHONDIR": cls.profile.name}):
            cls.shell = InteractiveShell.instance(config=config)
        cls.shell.run_line_magic("load_ext", "load_clean")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="notebook-magic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.helper = self.root / "magic_helper.py"
        self.helper.write_text("def double(value):\n    return value * 2\n", encoding="utf-8")
        sys.path.insert(0, str(self.root))
        self.addCleanup(sys.path.remove, str(self.root))
        sys.modules.pop("magic_helper", None)
        self.addCleanup(sys.modules.pop, "magic_helper", None)
        importlib.invalidate_caches()
        self.shell.payload_manager.clear_payload()
        self.shell.rl_next_input = ""

    def run_line(self, name, args=""):
        with redirect_stdout(StringIO()) as output:
            self.shell.run_line_magic(name, args)
        return output.getvalue()

    def run_cell(self, name, args, source):
        with redirect_stdout(StringIO()) as output:
            self.shell.run_cell_magic(name, args, source)
        return output.getvalue()

    def test_extension_registers_all_magics(self):
        for name in ["load_clean", "load_clean_preview", "load_clean_status",
                     "load_clean_check", "load_clean_materialize"]:
            self.assertIsNotNone(self.shell.find_line_magic(name), name)
        for name in ["load_clean", "load_clean_save"]:
            self.assertIsNotNone(self.shell.find_cell_magic(name), name)

    def test_default_and_inline_import_execution(self):
        for mode in ["default", "inline"]:
            with self.subTest(mode=mode):
                self.run_cell("load_clean", mode, "from magic_helper import double")
                self.assertEqual(self.shell.user_ns["double"](3), 6)

    def test_preview_check_status_and_materialize(self):
        self.run_cell("load_clean", "default", "from magic_helper import double")
        self.assertIn("magic_helper", self.run_line("load_clean_preview", "default from magic_helper import double"))
        self.assertIn("valid", self.run_line("load_clean_check", "default from magic_helper import double"))
        self.assertIn("unchanged", self.run_line("load_clean_status"))
        self.run_line("load_clean_materialize", "inline from magic_helper import double")
        self.assertIn("def double", self.shell.rl_next_input)

    def test_save_preview_and_creation_do_not_execute_body(self):
        path = self.root / "new module.py"
        arg = shlex.quote(path.as_posix())
        source = "def created(value):\n    return value + 1\n"
        self.run_cell("load_clean_save", f"--preview {arg}", source)
        self.assertFalse(path.exists())
        self.shell.user_ns.pop("created", None)
        self.run_cell("load_clean_save", arg, source)
        self.assertEqual(path.read_text(), source)
        self.assertNotIn("created", self.shell.user_ns)

    def test_linked_load_and_cell_sync(self):
        arg = shlex.quote(self.helper.as_posix())
        before = self.helper.read_bytes()
        self.run_line("load_clean", arg)
        self.assertEqual(self.helper.read_bytes(), before)
        self.assertIn("%%load_clean", self.shell.rl_next_input)
        baseline = fingerprint(self.helper.read_text())
        self.run_cell("load_clean", f"{arg} --base {baseline}",
                      "def double(value):\n    return value * 3\n")
        self.assertIn("value * 3", self.helper.read_text())
        self.assertEqual(self.shell.user_ns["double"](3), 9)

    def test_import_link_cell(self):
        self.run_cell("load_clean", "", "from magic_helper import double")
        self.assertEqual(self.shell.user_ns["double"](4), 8)
        self.assertIn("%%load_clean", self.shell.rl_next_input)


if __name__ == "__main__":
    unittest.main()
