"""The recorded phrase set (``bench/phrases.toml``): what each speaker says, and its true label."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from stride.intent import Intent, IntentKind


class PhraseGroup(StrEnum):
    CORE = "core"  # read as written; may be trained on
    HELDOUT = "heldout"  # read as written; never trained on
    OWN_WORDS = "own_words"  # speaker phrases a situation; never trained on


class PhraseError(ValueError):
    pass


@dataclass(frozen=True)
class Phrase:
    id: str
    group: PhraseGroup
    label: Intent
    text: str | None = None  # what to read aloud (core / heldout)
    prompt: str | None = None  # the situation to describe (own_words)
    post_rep: bool = False

    @property
    def trainable(self) -> bool:
        return self.group is PhraseGroup.CORE

    @property
    def display(self) -> str:
        """What the recorder shows the speaker."""
        return self.text if self.text is not None else f"In your own words: {self.prompt}"


_KEYS = {"id", "text", "prompt", "intent", "seconds", "reps", "distance_m", "group", "post_rep"}


def _phrase(raw: dict[str, Any]) -> Phrase:
    pid = raw.get("id")
    if not isinstance(pid, str) or not pid:
        raise PhraseError(f"phrase without an id: {raw}")
    if extra := set(raw) - _KEYS:
        raise PhraseError(f"{pid}: unknown keys {sorted(extra)}")
    try:
        group = PhraseGroup(raw["group"])
        kind = IntentKind(raw["intent"])
    except (KeyError, ValueError) as e:
        raise PhraseError(f"{pid}: bad or missing group/intent ({e})") from e
    text, prompt = raw.get("text"), raw.get("prompt")
    if group is PhraseGroup.OWN_WORDS:
        if prompt is None or text is not None:
            raise PhraseError(f"{pid}: own_words phrases need a prompt and no text")
    elif text is None or prompt is not None:
        raise PhraseError(f"{pid}: {group} phrases need text and no prompt")
    slots = {k: raw[k] for k in ("seconds", "reps", "distance_m") if k in raw}
    if any(not isinstance(v, int) or v <= 0 for v in slots.values()):
        raise PhraseError(f"{pid}: slots must be positive integers: {slots}")
    if kind is not IntentKind.PLAN and ({"reps", "distance_m"} & set(slots)):
        raise PhraseError(f"{pid}: reps/distance_m only apply to plans")
    return Phrase(
        id=pid,
        group=group,
        label=Intent(kind, **slots),
        text=text,
        prompt=prompt,
        post_rep=bool(raw.get("post_rep", False)),
    )


def load_phrases(path: Path) -> tuple[int, list[Phrase]]:
    """Return ``(version, phrases)``. Raises :class:`PhraseError` on any malformed entry."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    version = data.get("version")
    if not isinstance(version, int):
        raise PhraseError("phrases file needs an integer `version`")
    phrases = [_phrase(raw) for raw in data.get("phrase", [])]
    ids = [p.id for p in phrases]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise PhraseError(f"duplicate phrase ids: {sorted(dupes)}")
    return version, phrases


def to_web_json(version: int, phrases: list[Phrase]) -> dict[str, Any]:
    """The phrase set as the recording page reads it (``docs/record/phrases.json``)."""
    return {
        "version": version,
        "phrases": [
            {
                "id": p.id,
                "group": p.group.value,
                "display": p.display,
                "own_words": p.group is PhraseGroup.OWN_WORDS,
                "post_rep": p.post_rep,
            }
            for p in phrases
        ],
    }


if __name__ == "__main__":  # regenerate: python -m stride.bench.phrases > docs/record/phrases.json
    import json

    print(json.dumps(to_web_json(*load_phrases(Path("bench/phrases.toml"))), indent=1))
