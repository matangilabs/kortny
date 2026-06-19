"""SKILL.md model + parser contract (HIG-281 ADK removal).

Pins the validation behavior of the in-house skill models so swapping off
``google.adk.skills.models`` is provably behavior-preserving: community
Claude/Codex/ADK skills must still parse (or fail) exactly as before.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from kortny.skills.ingestion import SkillIngestionError, _parse_skill_markdown
from kortny.skills.skill_models import Frontmatter


class TestFrontmatter:
    def test_minimal_valid(self) -> None:
        fm = Frontmatter(name="my-skill", description="Does a thing.")
        assert fm.name == "my-skill"
        assert fm.allowed_tools is None
        assert fm.metadata == {}

    def test_allowed_tools_alias_and_split(self) -> None:
        fm = Frontmatter.model_validate(
            {
                "name": "s",
                "description": "d",
                "allowed-tools": "web_search pdf_generator",
            }
        )
        assert (fm.allowed_tools or "").split() == ["web_search", "pdf_generator"]

    def test_extra_frontmatter_fields_allowed(self) -> None:
        fm = Frontmatter.model_validate(
            {"name": "s", "description": "d", "tags": "a, b", "custom": "x"}
        )
        assert fm.tags == "a, b"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "bad", ["Foo", "foo bar", "foo_bar", "-foo", "foo-", "foo--bar", "a" * 65]
    )
    def test_invalid_names_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            Frontmatter(name=bad, description="d")

    @pytest.mark.parametrize("good", ["foo", "foo-bar", "foo-bar-baz", "a" * 64, "v2"])
    def test_valid_names_accepted(self, good: str) -> None:
        assert Frontmatter(name=good, description="d").name == good

    def test_description_required_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            Frontmatter(name="s", description="")

    def test_description_length_cap(self) -> None:
        assert Frontmatter(name="s", description="x" * 1024).description
        with pytest.raises(ValidationError):
            Frontmatter(name="s", description="x" * 1025)

    def test_compatibility_length_cap(self) -> None:
        with pytest.raises(ValidationError):
            Frontmatter(name="s", description="d", compatibility="x" * 501)


class TestParseSkillMarkdown:
    def test_valid(self) -> None:
        md = (
            "---\nname: my-skill\ndescription: Does things\n"
            "allowed-tools: web_search pdf_generator\n---\nBody here."
        )
        fm, body = _parse_skill_markdown(md)
        assert fm.name == "my-skill"
        assert (fm.allowed_tools or "").split() == ["web_search", "pdf_generator"]
        assert body == "Body here."

    def test_missing_frontmatter(self) -> None:
        with pytest.raises(SkillIngestionError):
            _parse_skill_markdown("no frontmatter here")

    def test_unclosed_frontmatter(self) -> None:
        with pytest.raises(SkillIngestionError):
            _parse_skill_markdown("---\nname: s\ndescription: d\n")

    def test_frontmatter_not_a_mapping(self) -> None:
        with pytest.raises(SkillIngestionError):
            _parse_skill_markdown("---\n- a\n- b\n---\nbody")

    def test_malformed_yaml(self) -> None:
        with pytest.raises(SkillIngestionError):
            _parse_skill_markdown("---\nname: [unclosed\n---\nbody")

    def test_invalid_frontmatter_wrapped_as_ingestion_error(self) -> None:
        with pytest.raises(SkillIngestionError):
            _parse_skill_markdown("---\nname: Bad Name\ndescription: d\n---\nbody")


def test_skill_ingestion_does_not_import_google_adk() -> None:
    """Importing skill ingestion must not pull in google.adk (HIG-281).

    Run in a fresh interpreter so a sibling test that imports ADK can't pollute
    this process's sys.modules and mask a regression.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kortny.skills.ingestion, sys; "
            "assert 'google.adk' not in sys.modules, sorted(m for m in sys.modules if 'adk' in m)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
