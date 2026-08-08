from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitPageSmokeTests(unittest.TestCase):
    def test_new_pages_render_without_framework_runs_or_api_key(self) -> None:
        original = {name: os.environ.get(name) for name in ("DATABASE_PATH", "APP_PASSWORD", "OPENROUTER_API_KEY")}
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["DATABASE_PATH"] = str(Path(directory) / "ui.db")
                os.environ["APP_PASSWORD"] = ""
                os.environ.pop("OPENROUTER_API_KEY", None)
                expected = {
                    "pages/1_Run_Evaluation.py": "Run Evaluation",
                    "pages/2_Framework_Results.py": "Framework Results",
                    "pages/3_Prompt_Comparison.py": "Prompt Comparison",
                }
                for relative_path, title in expected.items():
                    app = AppTest.from_file(
                        ROOT / relative_path,
                        default_timeout=60,
                    ).run()
                    self.assertEqual(len(app.exception), 0, relative_path)
                    self.assertEqual([item.value for item in app.title], [title])
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
