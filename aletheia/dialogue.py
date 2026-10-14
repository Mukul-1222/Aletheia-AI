from __future__ import annotations

import random
import re
from dataclasses import dataclass


_TURN_RE = re.compile(r"^\s*([^:\n]{1,80}):\s*(.+?)\s*$")
_IGNORED_TIER_RE = re.compile(r"^\s*[%$][^:\n]*(?::\s*.*)?$")
_TRANSCRIPT_BOUNDARY_RE = re.compile(r"^\s*@(?:begin|end)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str


def format_turn_prompt(message: str, context: str = "") -> str:
    """Serialize an inference prompt using the same structural Turn A/Turn B format.

    Context is optional and remains separate from the current user turn. This is a
    formatting contract, not a semantic answer template.
    """
    user = str(message).strip()
    prefix = str(context).strip()
    turn = f"Turn A:\n{user}\nTurn B:\n"
    return f"{prefix}\n\n{turn}" if prefix else turn


def format_dialogue_example(prompt: str, response: str, context: str = "") -> str:
    """Serialize a supervised example, retaining observed multi-turn context.

    ``dialogue_response_pairs`` returns a serialized sequence of earlier observed
    turns, rather than a bare user utterance. Preserve those structural turns and
    append the response marker; never wrap a transcript inside another Turn A.
    """
    prefix = str(prompt).strip()
    if context.strip():
        prefix = f"{context.strip()}\n\n{prefix}" if prefix else context.strip()
    if "Turn A:" in prefix or "Turn B:" in prefix:
        return prefix + "\nTurn B:\n" + str(response).strip() + "\n"
    return format_turn_prompt(prefix) + str(response).strip() + "\n"


def parse_dialogue_example(text: str) -> tuple[str, str] | None:
    """Parse the canonical turn markers without guessing when boundaries are absent."""
    raw = str(text)
    if "Turn A:" not in raw or "Turn B:" not in raw:
        return None
    after_a = raw.split("Turn A:", 1)[1]
    if "Turn B:" not in after_a:
        return None
    prompt = after_a.split("Turn B:", 1)[0].strip()
    response = after_a.split("Turn B:", 1)[1].strip()
    return (prompt, response) if prompt and response else None


def _clean_speaker(raw: str) -> str | None:
    speaker = " ".join(raw.split())[:80]
    if not speaker or len(speaker.split()) > 5:
        return None
    # CHILDES and similar transcript formats use dependent/annotation tiers
    # such as %mor and %int. They are not conversational speakers.
    if speaker.startswith(("%", "@", "$")):
        return None
    if any(char in speaker for char in ".!?;[]{}"):
        return None
    return speaker


def extract_dialogue_turns(text: str, *, min_turn_chars: int = 8) -> list[list[DialogueTurn]]:
    """Extract structurally consistent speaker-labelled runs without semantic labels.

    Formatting and CHILDES metadata/dependent tiers are handled structurally rather than
    treated as speakers. Blank lines, ``@Begin`` and ``@End`` delimit observed runs, while
    ``%...`` and ``$...`` tiers are ignored without breaking a run. Other ``@...`` headers
    are ignored without breaking a run.

    Consecutive utterances from the same speaker are merged into one structural turn before
    validation. Real transcripts frequently contain several adjacent turns from one speaker;
    requiring strict alternation would otherwise discard otherwise valid conversations.
    """
    parsed: list[DialogueTurn | None] = []
    speaker_counts: dict[str, int] = {}
    pending_blank = False
    current_first_speaker: str | None = None
    current_has_multiple_speakers = False

    for raw_line in text.splitlines():
        if not raw_line.strip():
            pending_blank = True
            continue
        if _TRANSCRIPT_BOUNDARY_RE.match(raw_line):
            parsed.append(None)
            pending_blank = False
            current_first_speaker = None
            current_has_multiple_speakers = False
            continue
        if _IGNORED_TIER_RE.match(raw_line) or raw_line.lstrip().startswith("@"):
            # Dependent tiers and ordinary @ metadata do not terminate an observed run.
            continue
        match = _TURN_RE.match(raw_line)
        if not match:
            parsed.append(None)
            pending_blank = False
            current_first_speaker = None
            current_has_multiple_speakers = False
            continue
        speaker = _clean_speaker(match.group(1))
        body = " ".join(match.group(2).split())
        if speaker is None or len(body) < max(1, int(min_turn_chars)):
            parsed.append(None)
            pending_blank = False
            current_first_speaker = None
            current_has_multiple_speakers = False
            continue

        # A blank line normally acts as formatting. Treat it as a soft transcript boundary
        # only when a complete-looking two-speaker exchange is followed by that exchange's
        # first participant again. This keeps Human/Assistant transcripts with visual spacing
        # intact while allowing separate A/B records to become distinct holdout runs.
        if pending_blank and current_first_speaker is not None and current_has_multiple_speakers and speaker == current_first_speaker:
            parsed.append(None)
            current_first_speaker = None
            current_has_multiple_speakers = False

        parsed.append(DialogueTurn(speaker=speaker, text=body))
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if current_first_speaker is None:
            current_first_speaker = speaker
        elif speaker != current_first_speaker:
            current_has_multiple_speakers = True
        pending_blank = False

    runs: list[list[DialogueTurn]] = []
    current: list[DialogueTurn] = []

    def flush() -> None:
        nonlocal current
        if len(current) < 2:
            current = []
            return

        # Merge adjacent utterances from the same participant. This preserves the observed
        # local ordering without inventing intent or roles and prevents natural transcripts
        # such as A, A, B from being discarded as malformed A/B alternation.
        original_turn_count = len(current)
        merged: list[DialogueTurn] = []
        for turn in current:
            if merged and merged[-1].speaker == turn.speaker:
                merged[-1] = DialogueTurn(
                    speaker=turn.speaker,
                    text=f"{merged[-1].text} {turn.text}".strip(),
                )
            else:
                merged.append(turn)
        current = merged

        if len(current) < 2:
            current = []
            return

        run_counts: dict[str, int] = {}
        for turn in current:
            run_counts[turn.speaker] = run_counts.get(turn.speaker, 0) + 1
        if len(run_counts) < 2:
            current = []
            return

        ordered = [turn.speaker for turn in current]
        if len(run_counts) == 2:
            # Keep the conservative validation against isolated prose headings. After
            # consecutive same-speaker turns are merged, genuine two-speaker transcripts
            # naturally reduce to alternating speaker changes.
            alternating = all(ordered[i] == ordered[i - 2] for i in range(2, len(ordered)))
            repeated_labels = all(speaker_counts.get(label, 0) >= 2 for label in run_counts)
            if alternating and (original_turn_count >= 3 or repeated_labels):
                runs.append(current)
        else:
            # Multi-party conversations do not have a meaningful A/B alternation. It is
            # enough that validated turns change participant after same-speaker merging.
            changes_speaker = all(ordered[i] != ordered[i - 1] for i in range(1, len(ordered)))
            if len(current) >= 3 and changes_speaker:
                runs.append(current)
        current = []

    for item in parsed:
        if item is None:
            if current:
                flush()
            continue
        current.append(item)

    if current:
        flush()
    return runs


def _pairs_from_run(run: list[DialogueTurn]) -> list[tuple[str, str]]:
    """Derive forward context/response pairs from one validated dialogue run."""
    speakers = []
    for turn in run:
        if turn.speaker not in speakers:
            speakers.append(turn.speaker)

    pairs: list[tuple[str, str]] = []
    if len(speakers) == 2:
        mapping = {speakers[0]: "A", speakers[1]: "B"}
        normalized = [(mapping[t.speaker], t.text) for t in run]
        for idx, (speaker, _text_value) in enumerate(normalized):
            if speaker != "B":
                continue
            context_start = max(0, idx - 3)
            context = normalized[context_start:idx]
            prompt_lines = [f"Turn {side}:\n{body}" for side, body in context]
            prompt = "\n".join(prompt_lines).strip()
            response = normalized[idx][1].strip()
            if prompt and response:
                pairs.append((prompt, response))
        return pairs

    # For multi-party runs, every turn that follows a different speaker provides a
    # valid observed context/response event. Labels remain structural only.
    for idx in range(1, len(run)):
        if run[idx].speaker == run[idx - 1].speaker:
            continue
        response_speaker = run[idx].speaker
        mapping = {response_speaker: "B"}
        next_labels = iter([label for label in "ACDEFGHIJKLMNOPQRSTUVWXYZ"])
        for speaker in speakers:
            if speaker == response_speaker:
                continue
            mapping[speaker] = next(next_labels)
        normalized = [(mapping[t.speaker], t.text) for t in run]
        context_start = max(0, idx - 3)
        context = normalized[context_start:idx]
        prompt_lines = [f"Turn {side}:\n{body}" for side, body in context]
        prompt = "\n".join(prompt_lines).strip()
        response = normalized[idx][1].strip()
        if prompt and response:
            pairs.append((prompt, response))
    return pairs


def _limit_pairs(
    pairs: list[tuple[str, str]],
    max_examples: int,
    *,
    rng: random.Random,
) -> list[tuple[str, str]]:
    if max_examples <= 0 or len(pairs) <= max_examples:
        return pairs
    return rng.sample(pairs, int(max_examples))


def split_dialogue_response_pairs(
    text: str,
    *,
    holdout_fraction: float = 0.20,
    max_train_examples: int = 0,
    max_eval_examples: int = 128,
    seed: int = 1337,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split dialogue by whole transcript runs before deriving response pairs."""
    runs = extract_dialogue_turns(text, min_turn_chars=1)
    if len(runs) < 2 or holdout_fraction <= 0:
        train_runs, eval_runs = runs, []
    else:
        rng = random.Random(int(seed))
        order = list(range(len(runs)))
        rng.shuffle(order)
        eval_count = max(1, int(round(len(runs) * float(holdout_fraction))))
        eval_count = min(eval_count, len(runs) - 1)
        eval_ids = set(order[:eval_count])
        train_runs = [runs[i] for i in range(len(runs)) if i not in eval_ids]
        eval_runs = [runs[i] for i in range(len(runs)) if i in eval_ids]

    train_pairs = [pair for run in train_runs for pair in _pairs_from_run(run)]
    eval_pairs = [pair for run in eval_runs for pair in _pairs_from_run(run)]
    train_rng = random.Random(int(seed) + 1)
    eval_rng = random.Random(int(seed) + 2)
    train_pairs = _limit_pairs(train_pairs, int(max_train_examples), rng=train_rng)
    eval_pairs = _limit_pairs(eval_pairs, int(max_eval_examples), rng=eval_rng)
    return train_pairs, eval_pairs


def dialogue_response_pairs(
    text: str,
    *,
    max_examples: int = 2000,
    seed: int | None = None,
) -> list[tuple[str, str]]:
    """Return randomized structural context/response pairs without semantic labels.

    The whole transcript is parsed before limiting the sample. Blank lines are preserved
    because they may delimit independent transcript runs.
    """
    pairs: list[tuple[str, str]] = []
    for run in extract_dialogue_turns(str(text), min_turn_chars=1):
        pairs.extend(_pairs_from_run(run))
    rng = random.Random(seed) if seed is not None else random.Random()
    return _limit_pairs(pairs, int(max_examples), rng=rng)


def dialogue_training_sequences(
    text: str,
    *,
    max_examples: int = 2000,
    seed: int | None = None,
) -> list[str]:
    """Build forward Turn A -> Turn B training strings from structural dialogue."""
    return [
        format_dialogue_example(prompt, response)
        for prompt, response in dialogue_response_pairs(text, max_examples=max_examples, seed=seed)
    ]
