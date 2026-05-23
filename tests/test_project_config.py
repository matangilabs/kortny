from pathlib import Path
from typing import Any

import tomli


def test_project_name_is_kortny() -> None:
    pyproject: dict[str, Any] = tomli.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["name"] == "kortny"
