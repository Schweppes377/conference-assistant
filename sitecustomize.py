"""Project-local TLS setup activated only by run-direct.ps1."""

from __future__ import annotations

import os


if os.getenv("CONFERENCE_ASSISTANT_DIRECT") == "1":
    import truststore

    # httpx/huggingface_hub otherwise use certifi and cannot validate the TLS
    # chain presented while Happ is active. Keep verification enabled and use
    # the same trusted Windows certificate store as native applications.
    truststore.inject_into_ssl()
