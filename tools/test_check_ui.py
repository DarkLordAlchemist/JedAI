#!/usr/bin/env python3
"""setup_lite.check_ui: every install checks that the web app (Chat, Monitor, About) is all there before it starts."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup_lite  # noqa: E402


class TestCheckUi(unittest.TestCase):
    def test_this_folder_has_the_whole_app(self):
        self.assertEqual(setup_lite.check_ui(ROOT), [])

    def test_missing_files_and_tabs_are_named(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            web = tmp / "serve" / "web"
            shutil.copytree(ROOT / "serve" / "web", web)
            (web / "monitor.html").unlink()
            (web / "index.html").write_text('<button data-tab="chat"></button>', encoding="utf-8")
            problems = setup_lite.check_ui(tmp)
            self.assertIn("serve/web/monitor.html", problems)
            self.assertIn("the Monitor tab", problems)
            self.assertIn("the About tab", problems)
            self.assertNotIn("the Chat tab", problems)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_folder(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            self.assertEqual(len(setup_lite.check_ui(tmp)), len(setup_lite.UI_FILES))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
