"""Dependency-graph invariants that only bite outside a developer machine.

These are packaging assertions rather than behaviour tests. They exist because
the failures they catch are silent locally and fatal in a container.
"""

from __future__ import annotations

from importlib.metadata import distributions
from pathlib import Path
import tomllib

import pytest

pytestmark = pytest.mark.unit

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def installed_distributions() -> set[str]:
    """Names of every installed distribution, lowercased."""
    return {name.lower() for dist in distributions() if (name := dist.metadata["Name"]) is not None}


def test_only_the_headless_opencv_build_is_installed() -> None:
    """`opencv-python` and `opencv-python-headless` both own the `cv2` package.

    Installing both is a race over which one lands on disk, and the GUI build
    links libGL and libxcb, which a slim image does not carry. Importing it
    there fails with ``libxcb.so.1: cannot open shared object file``.
    """
    installed = installed_distributions()

    assert "opencv-python-headless" in installed
    assert "opencv-python" not in installed, (
        "the GUI OpenCV build is installed; it shadows opencv-python-headless "
        "and cannot import in the Docker image"
    )


def test_cv2_imports_without_gui_libraries() -> None:
    import cv2

    assert cv2.__version__


def test_the_gui_opencv_build_is_overridden_out_of_the_graph() -> None:
    """The override is what keeps ultralytics from reintroducing the GUI build."""
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    overrides = manifest["tool"]["uv"]["override-dependencies"]

    assert any(entry.startswith("opencv-python ") for entry in overrides)


def test_the_declared_opencv_dependency_is_headless() -> None:
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = manifest["project"]["dependencies"]

    assert any(dep.startswith("opencv-python-headless") for dep in dependencies)
    assert not any(dep.startswith("opencv-python>") for dep in dependencies)
