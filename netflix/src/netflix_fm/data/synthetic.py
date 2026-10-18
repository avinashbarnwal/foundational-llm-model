"""Synthetic catalog and member histories, so the pipeline runs without real data."""

from __future__ import annotations

import random

from netflix_fm.config import VocabConfig
from netflix_fm.data.schema import Interaction, ItemMetadata

START_TS = 1_700_000_000
DAY = 86400


def make_catalog(vocab: VocabConfig, seed: int = 0) -> list[ItemMetadata]:
    rng = random.Random(seed)
    items = []
    for item_id in range(1, vocab.num_items + 1):
        items.append(
            ItemMetadata(
                item_id=item_id,
                genre=rng.randint(1, vocab.num_genres - 1),
                language=rng.randint(1, vocab.num_languages - 1),
                release_timestamp=START_TS - rng.randint(0, 3 * 365) * DAY,
                tags=rng.sample(range(1, vocab.num_metadata_tags), k=rng.randint(1, 6)),
            )
        )
    return items


def make_histories(
    vocab: VocabConfig,
    catalog: list[ItemMetadata],
    num_members: int = 200,
    events_per_member: tuple[int, int] = (50, 600),
    seed: int = 0,
) -> list[list[Interaction]]:
    """Members have a few favourite genres; sessions often produce several events per title."""
    rng = random.Random(seed)
    by_genre: dict[int, list[int]] = {}
    for item in catalog:
        by_genre.setdefault(item.genre, []).append(item.item_id)
    genres = sorted(by_genre)

    histories = []
    for _ in range(num_members):
        favourites = rng.sample(genres, k=min(3, len(genres)))
        device = rng.randint(1, vocab.num_devices - 1)
        locale = rng.randint(1, vocab.num_locales - 1)
        ts = START_TS + rng.randint(0, 30) * DAY
        events: list[Interaction] = []
        target = rng.randint(*events_per_member)
        while len(events) < target:
            genre = rng.choice(favourites) if rng.random() < 0.8 else rng.choice(genres)
            item_id = rng.choice(by_genre[genre])
            # One session on a title: a few play/pause segments close together.
            for _ in range(rng.randint(1, 4)):
                duration = rng.randint(30, 3600)
                events.append(
                    Interaction(
                        item_id=item_id,
                        timestamp=ts,
                        action_type=rng.randint(1, vocab.num_action_types - 1),
                        duration_seconds=duration,
                        device=device if rng.random() < 0.9 else rng.randint(1, vocab.num_devices - 1),
                        locale=locale,
                    )
                )
                ts += duration + rng.randint(0, 600)
            ts += rng.randint(3600, 3 * DAY)
        histories.append(events)
    return histories
