"""T1/T5/T8: thin-entrypoint contract -- no duplicated logic here."""

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


def test_smoke_kernel_wires_package_seams():
    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "notebooks" / "smoke_kernel.py").read_text(encoding="utf-8")
    assert "from restore.smoke import" in text
    assert "from restore.models import" in text
    assert "run_smoke" in text
    assert "git" in text and "clone" in text
    assert "def run_smoke(" not in text
    assert "def smoke_verdict(" not in text
    assert "def simulate_damage(" not in text
    assert "class DetectorNet" not in text
    assert "class RestorerNet" not in text


def test_demo_app_wires_package_seams():
    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "demo" / "app.py").read_text(encoding="utf-8")
    assert "from restore.demo import" in text
    assert "serve_restoration" in text
    assert "build_example" in text
    assert "def serve_restoration(" not in text
    assert "def restore(" not in text
    assert "def restore_tiled(" not in text
    assert "def detect(" not in text
    assert "def composite_output(" not in text
