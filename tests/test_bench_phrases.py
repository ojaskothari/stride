from __future__ import annotations

import re
from pathlib import Path

import pytest

from stride.bench.phrases import Phrase, PhraseError, PhraseGroup, load_phrases, to_web_json
from stride.intent import IntentKind, parse

PHRASES = Path(__file__).resolve().parents[1] / "bench" / "phrases.toml"
_, ALL = load_phrases(PHRASES)
CORE = [p for p in ALL if p.group is PhraseGroup.CORE]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


@pytest.mark.parametrize("phrase", CORE, ids=[p.id for p in CORE])
def test_core_labels_match_the_parser(phrase: Phrase) -> None:
    # A core phrase read perfectly must parse to its label; otherwise the label (or parser) is
    # wrong, and every clip of it would be scored against a bad answer.
    assert phrase.text is not None
    assert parse(phrase.text) == phrase.label


def test_heldout_phrasings_are_not_core_phrasings() -> None:
    core = {_norm(p.text or "") for p in CORE}
    for p in ALL:
        if p.group is PhraseGroup.HELDOUT:
            assert _norm(p.text or "") not in core, p.id


def test_every_command_intent_is_covered_in_core() -> None:
    covered = {p.label.kind for p in CORE}
    assert covered == set(IntentKind)


def test_post_rep_list_is_short_enough_for_practice() -> None:
    # About 4 phrases after each rep of an 8-rep session.
    assert 12 <= sum(p.post_rep for p in ALL) <= 24


def test_only_core_is_trainable() -> None:
    assert all(p.trainable == (p.group is PhraseGroup.CORE) for p in ALL)


@pytest.mark.parametrize(
    "body",
    [
        'id = "a"\ntext = "x"\nintent = "nope"\ngroup = "core"',
        'id = "a"\nprompt = "x"\nintent = "done"\ngroup = "core"',
        'id = "a"\ntext = "x"\nintent = "done"\ngroup = "own_words"',
        'id = "a"\ntext = "x"\nintent = "start_rest"\nreps = 3\ngroup = "core"',
        'id = "a"\ntext = "x"\nintent = "start_rest"\nseconds = 0\ngroup = "core"',
        'id = "a"\ntext = "x"\nintent = "done"\ngroup = "core"\ntypo = 1',
    ],
)
def test_malformed_phrases_are_rejected(tmp_path: Path, body: str) -> None:
    f = tmp_path / "p.toml"
    f.write_text(f"version = 1\n[[phrase]]\n{body}\n")
    with pytest.raises(PhraseError):
        load_phrases(f)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    entry = '[[phrase]]\nid = "a"\ntext = "go"\nintent = "go"\ngroup = "core"\n'
    f = tmp_path / "p.toml"
    f.write_text("version = 1\n" + entry + entry)
    with pytest.raises(PhraseError, match="duplicate"):
        load_phrases(f)


def test_recording_page_phrases_are_in_sync() -> None:
    import json

    page = PHRASES.parents[1] / "docs" / "record" / "phrases.json"
    expected = to_web_json(*load_phrases(PHRASES))
    assert json.loads(page.read_text()) == expected, (
        "regenerate: python -m stride.bench.phrases > docs/record/phrases.json"
    )
