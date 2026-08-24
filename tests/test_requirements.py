"""requirements.txt is a deliverable, not a scratch file.

A jury member clones this repo onto whatever Python python.org offers that
month. If that is a version the scientific stack has no wheels for yet, pip
silently falls back to compiling pandas/torch/scikit-learn from source and dies
inside Meson with "Could not find vswhere.exe" — an error that says nothing
about the actual cause. The guard lines at the top of requirements.txt turn
that into an immediate, readable refusal.

These tests exist so nobody deletes the guard while tidying up, and so no new
dependency lands without an upper bound.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"

#: Interpreters the project claims to support, and those it must refuse.
SUPPORTED = ("3.11", "3.12", "3.13", "3.14")
UNSUPPORTED = ("3.8", "3.9", "3.10", "3.15", "3.16", "4.0")


def _requirements() -> list[Requirement]:
    lignes = []
    for brute in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        ligne = brute.split("#")[0].strip()
        if ligne:
            lignes.append(Requirement(ligne))
    return lignes


def _is_guard(req: Requirement) -> bool:
    return req.name.lower().startswith("neuroreponse-requires-python")


def test_every_line_is_a_valid_requirement() -> None:
    """A typo here is invisible until someone installs. Parse the whole file."""
    reqs = _requirements()
    assert len(reqs) > 15, "requirements.txt looks truncated"


@pytest.mark.parametrize("version", SUPPORTED)
def test_supported_python_installs_the_real_dependencies(version: str) -> None:
    """On 3.11-3.14 no guard fires and every real dependency is selected."""
    env = {"python_version": version}
    actifs = [r for r in _requirements() if r.marker is None or r.marker.evaluate(env)]

    assert not any(_is_guard(r) for r in actifs), (
        f"the guard wrongly blocks Python {version}, which the project supports"
    )
    noms = {r.name.lower() for r in actifs}
    for paquet in ("numpy", "scipy", "pandas", "torch", "streamlit", "mne"):
        assert paquet in noms, f"{paquet} is not installed on Python {version}"


@pytest.mark.parametrize("version", UNSUPPORTED)
def test_unsupported_python_is_refused_on_the_first_line(version: str) -> None:
    """The guard must fire *before* pip ever reaches pandas.

    Order matters: pip collects requirements in file order, so the guard only
    short-circuits the Meson build if it comes first.
    """
    env = {"python_version": version}
    actifs = [r for r in _requirements() if r.marker is None or r.marker.evaluate(env)]

    assert actifs, f"nothing at all is selected on Python {version}"
    assert _is_guard(actifs[0]), (
        f"Python {version} is unsupported but the first requirement selected is "
        f"{actifs[0].name!r} — pip would start installing before it refuses"
    )


def test_the_guard_can_never_be_satisfied() -> None:
    """A guard that resolves to a real package is not a guard.

    ``== 0`` is the second half of the trick: even if someone were to register
    the name on PyPI, there is no version 0 to install.
    """
    for req in _requirements():
        if _is_guard(req):
            assert str(req.specifier) == "==0", (
                f"{req.name} must pin ==0 so it cannot resolve, got {req.specifier}"
            )


def test_every_dependency_has_an_upper_bound() -> None:
    """An uncapped dependency is a future breakage with no failing test.

    pandas 3.0 landed while this project was pinned ``pandas>=2.1``; the next
    major will land the same way. Bounds are one major above what is tested.
    """
    sans_borne = [
        r.name
        for r in _requirements()
        if not _is_guard(r) and not any(s.operator in ("<", "<=", "==") for s in r.specifier)
    ]
    assert not sans_borne, f"dependencies with no upper bound: {sans_borne}"


def test_documented_python_range_matches_the_guard() -> None:
    """The README tells users which Python to install; the guard enforces it.

    If those two drift apart, users follow the README and hit the guard.
    """
    readme = (REQUIREMENTS.parent / "README.md").read_text(encoding="utf-8")
    assert "3.11 – 3.14" in readme or "3.11 - 3.14" in readme, (
        "README no longer states the supported Python range"
    )
    for req in _requirements():
        if _is_guard(req) and "too-new" in req.name:
            assert '>= "3.15"' in str(req.marker), (
                f"README says 3.11-3.14 but the guard fires at {req.marker}"
            )
