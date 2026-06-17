"""render.py — Write the self-contained demo HTML file.

Reads the ECharts driver template, injects the precomputed payload JSON
and the vendored ECharts library, writes a single offline HTML file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_VENDOR_DIR = Path(__file__).parent / "vendor"
_DEFAULT_ECHARTS = _VENDOR_DIR / "echarts.min.js"


def read_vendored_echarts(path: Path | None = None) -> str:
    """Return the vendored echarts.min.js content as a string.

    Parameters
    ----------
    path : Path | None
        Override path to echarts.min.js. Defaults to src/demo/vendor/echarts.min.js.

    Raises
    ------
    FileNotFoundError
        With a clear message including the one-line fetch command if the file
        is missing.
    """
    p = Path(path) if path is not None else _DEFAULT_ECHARTS
    if not p.exists():
        raise FileNotFoundError(
            f"Vendored ECharts not found at {p}.\n"
            "Fetch it once (requires internet):\n"
            "  curl -fsSL https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"
            f" -o {p}"
        )
    return p.read_text(encoding="utf-8")


def write_demo_html(
    payload: dict[str, Any],
    output_path: Path,
    echarts_js: str | None = None,
    echarts_path: Path | None = None,
) -> None:
    """Write a self-contained offline HTML demo file.

    Parameters
    ----------
    payload : dict
        Precomputed JSON payload from serialize.build_demo_payload().
    output_path : Path
        Destination file path.
    echarts_js : str | None
        Pre-read ECharts JS text. If None, reads from echarts_path or the
        default vendor location.
    echarts_path : Path | None
        Override path for read_vendored_echarts() when echarts_js is None.
    """
    if echarts_js is None:
        echarts_js = read_vendored_echarts(echarts_path)

    template_path = _TEMPLATES_DIR / "demo.html"
    template = template_path.read_text(encoding="utf-8")

    # Inline ECharts: replace the sentinel placeholder
    # (The template uses a {{ECHARTS_JS}} sentinel instead of a network URL)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    html = template.replace("{{ECHARTS_JS}}", echarts_js)
    html = html.replace("{{PAYLOAD_JSON}}", payload_json)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
