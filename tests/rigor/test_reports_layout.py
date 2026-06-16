"""tests/rigor/test_reports_layout.py

Enforce the reports/ tracking convention encoded in .gitignore so the curated
canonical record stays consistent and bloat-free as new reports are generated.

Convention (see the `reports/` allowlist block in .gitignore):
  - Only small summary artifacts are committed: .json / .md / .csv / static .png.
  - Regenerable bloat is NEVER tracked: *.html, *.npz, *.parquet, or anything
    under a plots/ directory.
  - Canonical run dirs use explicit names: {kind}_{arch}_{tf} for tuned /
    bootstrap carrier runs, baselines/ for the naive+dual-FVG floor, milestone
    date dirs, and one named experiment dir per sweep (tradesim_<date>/).

These tests read the git index (`git ls-files`), not the working tree, so they
assert what is actually committed — the thing the convention is about.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# Extensions / path fragments that must never appear in the tracked reports record.
_BLOAT_SUFFIXES = (".html", ".npz", ".parquet")
_BLOAT_DIR_FRAGMENT = "/plots/"

# Headline canonical artifacts that must stay tracked (the project's evidence base).
_REQUIRED_TRACKED = (
    "reports/rigor/baselines/naive_baselines.json",
    "reports/rigor/baselines/dual_fvg_compare.json",
    "reports/rigor/tune_cnn_lstm_5m/best_hp_cnn_lstm_5m.json",
    "reports/rigor/tune_cnn_lstm_15m/best_hp_cnn_lstm_15m.json",
    "reports/rigor/bootstrap_cnn_lstm_5m/bootstrap_ci_cnn_lstm_5m.json",
    "reports/rigor/bootstrap_cnn_lstm_15m/bootstrap_ci_cnn_lstm_15m.json",
    "reports/rigor/09-Jun-26/phase4_verdict.md",
    "reports/rigor/07-Jun-26/dl_diagnosis_decision.md",
    "reports/inspect/tradesim_2026-06-15/README.md",
)


def _tracked_reports_files() -> list[str]:
    """Return every path git currently tracks under reports/ (forward slashes)."""
    out = subprocess.run(
        ["git", "ls-files", "reports/"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=_REPO,
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not _git_available(), reason="git not available")


def test_no_regenerable_bloat_is_tracked():
    """No *.html / *.npz / *.parquet and nothing under plots/ may be committed."""
    offenders = [
        f
        for f in _tracked_reports_files()
        if f.endswith(_BLOAT_SUFFIXES) or _BLOAT_DIR_FRAGMENT in f
    ]
    assert not offenders, (
        "Regenerable bloat is tracked under reports/ — the .gitignore allowlist "
        f"should keep these out:\n  " + "\n  ".join(offenders)
    )


def test_only_summary_extensions_tracked():
    """Tracked reports are small summary artifacts: json / md / csv / static png."""
    allowed = (".json", ".md", ".csv", ".png")
    bad = [f for f in _tracked_reports_files() if not f.endswith(allowed)]
    assert not bad, (
        "Files with non-summary extensions are tracked under reports/ "
        f"(allowed: {allowed}):\n  " + "\n  ".join(bad)
    )


@pytest.mark.parametrize("rel", _REQUIRED_TRACKED)
def test_headline_canonical_artifacts_are_tracked(rel):
    """The headline evidence base must remain committed, not local-only."""
    assert rel in set(_tracked_reports_files()), (
        f"Canonical artifact not tracked: {rel}. If it moved, update both "
        ".gitignore and this test."
    )


def test_carrier_run_dirs_follow_naming_scheme():
    """tune_/bootstrap_ carrier dirs use the explicit {kind}_{arch}_{tf} scheme."""
    rigor = _REPO / "reports" / "rigor"
    for kind in ("tune", "bootstrap"):
        for tf in ("5m", "15m"):
            d = rigor / f"{kind}_cnn_lstm_{tf}"
            assert d.is_dir(), f"expected explicitly-named carrier dir: {d}"
