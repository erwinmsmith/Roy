from __future__ import annotations

import os
import subprocess
from pathlib import Path


LOADER = Path(__file__).parents[1] / "remote" / "load_roy_env.sh"


def _load(tmp_path: Path, content: str, inherited: str = "") -> str:
    (tmp_path / ".env").write_text(content, encoding="utf-8")
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = inherited
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; load_deepseek_api_key "$2"; printf "%s" "$DEEPSEEK_API_KEY"',
            "bash",
            str(LOADER),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout


def test_loads_unquoted_and_quoted_deepseek_key(tmp_path: Path) -> None:
    assert _load(tmp_path, "IGNORED=x\nDEEPSEEK_API_KEY=from-file\n") == "from-file"
    assert _load(tmp_path, 'DEEPSEEK_API_KEY="quoted-value"\n') == "quoted-value"


def test_exported_deepseek_key_takes_precedence(tmp_path: Path) -> None:
    assert _load(tmp_path, "DEEPSEEK_API_KEY=from-file\n", "from-shell") == "from-shell"
