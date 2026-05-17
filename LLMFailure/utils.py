"""
Thin wrapper around the OpenRouter chat-completions API used by every
`*_evaluate.py` script in this repository.

The API key must be provided via the `OPENROUTER_API_KEY` environment variable.
"""
from __future__ import annotations

import json
import os
from time import sleep

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running, e.g.:\n"
            "    export OPENROUTER_API_KEY='sk-or-v1-...'"
        )
    return key


def prompt_model(
    prompt: str,
    model_name: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    """Send a single chat-completion request and return the response text.

    Retries indefinitely on transient request failures (network errors, etc.)
    with a 5-second backoff, matching the original behavior the evaluators
    were written against.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "system", "content": system_prompt},
        ],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {_get_api_key()}"}

    while True:
        try:
            response = requests.post(
                url=OPENROUTER_URL,
                headers=headers,
                data=json.dumps(payload),
            )
            break
        except requests.RequestException:
            print(f"pipeline broke, trying again... {model_name}")
            sleep(5)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, ValueError):
        print("ERRRRRR", response.text)
        return ""
