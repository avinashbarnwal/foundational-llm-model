import pytest

from netflix_fm.config import Config, DataConfig, ModelConfig, ObjectiveConfig, TrainingConfig, VocabConfig
from netflix_fm.data.dataset import InteractionWindowDataset
from netflix_fm.data.features import ItemCatalog
from netflix_fm.data.synthetic import make_catalog, make_histories
from netflix_fm.data.tokenizer import InteractionTokenizer


@pytest.fixture
def cfg() -> Config:
    return Config(
        data=DataConfig(max_seq_len=32, sliding_window_stride=16),
        vocab=VocabConfig(num_items=200, num_metadata_tags=50),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=4, dropout=0.0, attention_window=8, global_tokens=2),
        objective=ObjectiveConfig(multi_token_horizon=2),
        training=TrainingConfig(batch_size=8, epochs=1),
    )


@pytest.fixture
def items(cfg):
    return make_catalog(cfg.vocab, seed=1)


@pytest.fixture
def catalog(cfg, items) -> ItemCatalog:
    return ItemCatalog(items, cfg.vocab.num_items)


@pytest.fixture
def sequences(cfg, items):
    histories = make_histories(cfg.vocab, items, num_members=12, events_per_member=(20, 120), seed=1)
    tok = InteractionTokenizer(cfg.data.merge_gap_seconds)
    return [tok.tokenize(h) for h in histories]


@pytest.fixture
def dataset(cfg, catalog, sequences) -> InteractionWindowDataset:
    return InteractionWindowDataset(
        sequences, catalog, cfg.data.max_seq_len, cfg.data.sliding_window_stride,
        cfg.objective.multi_token_horizon, cfg.data.min_watch_seconds,
    )
