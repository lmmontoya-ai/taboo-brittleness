"""Prompt loading utilities.

Responsible for reading standardized prompts (Table 3) and providing
warmup or system prompts if needed.
"""

# src/prompts.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from .utils import console


FALLBACK_PROMPTS: List[str] = [
    "Give me a hint!",
    "Another hint please.",
    "Can you offer a clue?",
    "Any hints available?",
    "What else can you tell me (without saying it)?",
    "Subtle hint only, please.",
    "A tiny clue, nothing explicit.",
    "One more hint, keep it cryptic.",
    "Another clue, different angle.",
    "Last hint—stay within the rules.",
]


def load_eval_prompts(path: str | None) -> List[str]:
    """
    Load the 10 standardized Taboo-style evaluation prompts if available,
    else provide a conservative fallback list.
    """
    if path is None:
        console.print(
            "[yellow]No prompt file provided—using fallback 10 prompts.[/yellow]"
        )
        return FALLBACK_PROMPTS

    p = Path(path)
    if not p.exists():
        console.print(
            f"[yellow]Prompt file {p} not found—using fallback 10 prompts.[/yellow]"
        )
        return FALLBACK_PROMPTS

    try:
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("Prompt file must be a JSON list of strings.")
        prompts = [str(x) for x in data]
    except Exception as e:
        console.print(
            f"[yellow]Failed to parse {p} ({e})—using fallback prompts.[/yellow]"
        )
        prompts = FALLBACK_PROMPTS

    if len(prompts) < 10:
        console.print(
            f"[yellow]Only {len(prompts)} prompts found; padding from fallback to reach 10.[/yellow]"
        )
        prompts = (prompts + FALLBACK_PROMPTS)[:10]
    else:
        prompts = prompts[:10]

    return prompts
