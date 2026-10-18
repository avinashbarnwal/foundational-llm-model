"""Interaction tokenization.

Raw histories are long and noisy: a single viewing session can show up as many
play / pause / resume events on the same title. Analogous to BPE merging frequent
byte pairs, we merge adjacent events on the same title into one token while keeping
the information that matters (total watch time, strongest action type, event count).
This shortens sequences, letting the model see a longer slice of history for the same
compute.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from netflix_fm.data.schema import Interaction, InteractionToken

# Higher value = stronger signal; used to pick the representative action for a merged token.
DEFAULT_ACTION_PRIORITY: dict[int, int] = {}


class InteractionTokenizer:
    def __init__(
        self,
        merge_gap_seconds: int = 1800,
        action_priority: dict[int, int] | None = None,
    ) -> None:
        self.merge_gap_seconds = merge_gap_seconds
        self.action_priority = action_priority or DEFAULT_ACTION_PRIORITY

    def _priority(self, action_type: int) -> int:
        return self.action_priority.get(action_type, action_type)

    def _can_merge(self, token: InteractionToken, last_end: int, event: Interaction) -> bool:
        return (
            event.item_id == token.item_id
            and event.timestamp - last_end <= self.merge_gap_seconds
        )

    def tokenize(self, events: Iterable[Interaction]) -> list[InteractionToken]:
        """Merge a member's events (any order) into a chronological token sequence."""
        ordered: Sequence[Interaction] = sorted(events, key=lambda e: e.timestamp)
        tokens: list[InteractionToken] = []
        last_end = 0
        for event in ordered:
            if tokens and self._can_merge(tokens[-1], last_end, event):
                tok = tokens[-1]
                tok.total_duration_seconds += event.duration_seconds
                tok.num_events += 1
                if self._priority(event.action_type) > self._priority(tok.action_type):
                    tok.action_type = event.action_type
                # Device/locale of the latest event reflect the member's current context.
                tok.device = event.device
                tok.locale = event.locale
            else:
                tokens.append(
                    InteractionToken(
                        item_id=event.item_id,
                        timestamp=event.timestamp,
                        action_type=event.action_type,
                        total_duration_seconds=event.duration_seconds,
                        num_events=1,
                        device=event.device,
                        locale=event.locale,
                    )
                )
            last_end = event.timestamp + event.duration_seconds
        return tokens
