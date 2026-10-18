import dataclasses

import torch

from netflix_fm.data.dataset import collate
from netflix_fm.data.features import ItemCatalog
from netflix_fm.data.synthetic import make_catalog
from netflix_fm.inference.embeddings import align_to_previous
from netflix_fm.inference.predictor import MemberSession, RequestContext
from netflix_fm.model.attention import sparse_causal_mask
from netflix_fm.model.foundation_model import FoundationModel
from netflix_fm.training.incremental import warm_start
from netflix_fm.training.losses import compute_loss


def _model(cfg, catalog):
    torch.manual_seed(0)
    model = FoundationModel(cfg)
    model.load_catalog(catalog, reference_timestamp=1_800_000_000)
    return model


def test_sparse_mask_pattern():
    pos = torch.arange(10)
    mask = sparse_causal_mask(pos, pos, window=3, global_tokens=2)
    assert mask[9, 0] and mask[9, 1]          # global tokens
    assert not mask[9, 5] and mask[9, 7]      # outside / inside local window
    assert not mask[3, 4]                     # causal


def test_forward_and_loss_backward(cfg, catalog, dataset):
    model = _model(cfg, catalog)
    batch = collate([dataset[i] for i in range(4)])
    out = model(batch)
    K, L = cfg.objective.multi_token_horizon, cfg.data.max_seq_len
    assert out["item_logits"].shape == (K, 4, L, cfg.vocab.num_items + 1)
    loss, metrics = compute_loss(out, batch, cfg.objective)
    loss.backward()
    assert torch.isfinite(loss) and "item_t+2" in metrics


def test_kv_cache_matches_full_forward(cfg, catalog, sequences):
    """Incremental decoding (with cache eviction) must equal a full sparse forward pass."""
    model = _model(cfg, catalog).eval()
    tokens = sequences[0][:20]
    request = RequestContext(timestamp=tokens[-1].timestamp + 3600, device=2, locale=3)

    full = MemberSession(model, catalog)
    full.observe(tokens, request)

    incremental = MemberSession(model, catalog)
    incremental.observe(tokens[:5], RequestContext(tokens[5].timestamp, tokens[5].device, tokens[5].locale))
    for i in range(5, len(tokens)):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        req = RequestContext(nxt.timestamp, nxt.device, nxt.locale) if nxt else request
        incremental.observe([tokens[i]], req)

    torch.testing.assert_close(incremental.member_embedding(), full.member_embedding(), atol=1e-5, rtol=1e-4)
    window, n_global = cfg.model.attention_window, cfg.model.global_tokens
    assert incremental.cache.layers[0].keys.shape[2] <= window + n_global
    assert len(incremental.recommend(k=5)) == 5


def test_warm_start_grows_catalog(cfg, catalog):
    old = _model(cfg, catalog)
    bigger = dataclasses.replace(cfg, vocab=dataclasses.replace(cfg.vocab, num_items=cfg.vocab.num_items + 20))
    new_catalog = ItemCatalog(make_catalog(bigger.vocab, seed=1), bigger.vocab.num_items)
    new = warm_start(old, bigger, new_catalog, reference_timestamp=1_800_000_000)
    n = cfg.vocab.num_items + 1
    old_w, new_w = old.embed.item.id_embedding.weight, new.embed.item.id_embedding.weight
    assert torch.equal(new_w[:n], old_w)
    assert new_w.shape[0] == bigger.vocab.num_items + 1
    assert new_w[n:].abs().sum() > 0


def test_procrustes_recovers_rotation():
    torch.manual_seed(0)
    old = torch.randn(100, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    aligned, _ = align_to_previous(old @ q, old)
    torch.testing.assert_close(aligned, old, atol=1e-4, rtol=1e-4)
