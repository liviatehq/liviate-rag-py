"""API key resolution: explicit constructor argument, else LIVIATE_API_KEY.

No other source (config files, keyring, etc.) is supported by design.
"""

from __future__ import annotations

import os

_ENV_VAR = "LIVIATE_API_KEY"


def resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return env_value
    raise ValueError(
        f"No API key provided. Pass api_key= explicitly or set the "
        f"{_ENV_VAR} environment variable."
    )
