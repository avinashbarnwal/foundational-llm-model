"""Raw interaction events and the tokens built from them.

Each token mixes two kinds of information:
  * action-level attributes (when / where / how the member engaged), and
  * content-level attributes (which title, plus its metadata).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Index 0 is reserved for padding in every categorical vocabulary.
PAD_ID = 0


@dataclass
class Interaction:
    """One raw event from the member's history."""

    item_id: int
    timestamp: int          # unix seconds
    action_type: int        # play, trailer, thumbs-up, add-to-list, ...
    duration_seconds: int   # watch time for plays, 0 for non-play actions
    device: int
    locale: int


@dataclass
class ItemMetadata:
    """Content-side attributes; available even for brand-new titles."""

    item_id: int
    genre: int
    language: int
    release_timestamp: int
    tags: list[int] = field(default_factory=list)


@dataclass
class InteractionToken:
    """One element of the model's input sequence (possibly several merged events)."""

    item_id: int
    timestamp: int              # start of the merged span
    action_type: int            # most significant action in the span
    total_duration_seconds: int # summed watch time across merged events
    num_events: int             # how many raw events were merged
    device: int
    locale: int
