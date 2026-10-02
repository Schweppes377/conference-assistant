"""Small helpers for predictable Unicode CLI output on Windows."""

from __future__ import annotations

import sys


def configure_utf8_console() -> None:
    """Use UTF-8 for CLI output while replacing unsupported PDF glyphs safely."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
