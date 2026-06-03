"""Structural tests: the repository has the files and config it should.

These run without the dataset or any training — they assert the project is wired up
correctly (files present, .gitignore protects the data, requirements pinned).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_source_files_exist():
    for rel in [
        "main.py",
        "src/data.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
        "app/demo_app.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
    ]:
        assert (ROOT / rel).exists(), f"missing expected file: {rel}"


def test_expected_directories_exist():
    for rel in ["src", "app", "tests", "figures", "data"]:
        assert (ROOT / rel).is_dir(), f"missing expected directory: {rel}"


def test_gitignore_protects_dataset():
    text = (ROOT / ".gitignore").read_text()
    assert "data/" in text, "data/ must be gitignored (the CSV is ~150MB)"
    assert ".gitkeep" in text, "data/.gitkeep should be kept"


def test_data_gitkeep_present():
    assert (ROOT / "data" / ".gitkeep").exists()


def test_requirements_are_pinned():
    lines = [
        ln.strip()
        for ln in (ROOT / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines, "requirements.txt has no packages"
    for ln in lines:
        assert "==" in ln, f"dependency not pinned: {ln}"
    pkgs = {ln.split("==")[0].lower() for ln in lines}
    for required in ["numpy", "pandas", "scikit-learn", "torch", "matplotlib"]:
        assert required in pkgs, f"missing required dependency: {required}"
