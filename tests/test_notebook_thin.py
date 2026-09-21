"""T1: thin-notebook contract -- no duplicated pipeline logic here."""

import pathlib


def test_notebook_imports_package_seam():
    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "notebooks" / "train.py").read_text(encoding="utf-8")
    assert "from restore import" in text
    assert "git" in text and "clone" in text
    assert "def restore(" not in text
    assert "def restore_masked(" not in text
    assert "def detect(" not in text
    assert "def composite_output(" not in text
