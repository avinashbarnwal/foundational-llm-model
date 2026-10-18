from netflix_fm.data.schema import Interaction
from netflix_fm.data.tokenizer import InteractionTokenizer


def _ev(item, ts, dur, action=1):
    return Interaction(item_id=item, timestamp=ts, action_type=action, duration_seconds=dur, device=1, locale=1)


def test_tokenizer_merges_adjacent_same_title():
    events = [_ev(5, 0, 600), _ev(5, 700, 300, action=3), _ev(7, 2000, 100), _ev(5, 10_000, 50)]
    tokens = InteractionTokenizer(merge_gap_seconds=1800).tokenize(events)
    assert [t.item_id for t in tokens] == [5, 7, 5]
    assert tokens[0].total_duration_seconds == 900
    assert tokens[0].num_events == 2
    assert tokens[0].action_type == 3


def test_tokenizer_splits_on_large_gap():
    events = [_ev(5, 0, 60), _ev(5, 60 + 4000, 60)]
    assert len(InteractionTokenizer(merge_gap_seconds=1800).tokenize(events)) == 2


def test_dataset_targets_are_shifted(dataset):
    sample = dataset[0]
    s, start = dataset.windows[0]
    items = dataset.encoded[s]["item_id"]
    valid = ~sample["padding_mask"]
    n = int(valid.sum())
    assert (sample["target_items"][0, : n] == items[start + 1 : start + 1 + n]).all()
    assert (sample["target_items"][1, : n - 1] == items[start + 2 : start + 1 + n]).all()
    assert (sample["target_weights"][:, ~valid] == 0).all()
