from pathlib import Path

from src.utils.paths import project_root, RAW_DIR, OUTPUTS_DIR
from src.utils.text_preprocess import basic_clean


def test_project_root_exists():
    """Repo root should resolve and exist."""
    root = project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "src").exists()


def test_paths_are_relative_to_repo():
    """Key project paths should be anchored under repo root."""
    root = project_root()
    assert str(RAW_DIR).startswith(str(root))
    assert str(OUTPUTS_DIR).startswith(str(root))


def test_basic_clean_returns_string():
    """basic_clean should run and return a non-empty string for normal text."""
    s = "I can't believe this movie is not good! <br /><br />"
    out = basic_clean(s)
    assert isinstance(out, str)
    assert len(out) > 0


def test_basic_clean_keeps_negation_word_not():
    """
    Ensuring negation is preserved.
    """
    s = "I can't recommend it."
    out = basic_clean(s)
    assert "not" in out.split()