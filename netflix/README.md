# Netflix-style recommendation foundation model

A PyTorch reference implementation of the ideas in the Netflix Tech Blog post
[Foundation Model for Personalized Recommendation](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39):
one large autoregressive transformer trained on members' full interaction histories,
whose predictions and embeddings serve many downstream recommenders.

## Layout

```
netflix/
├── configs/default.yaml           # all hyper-parameters
├── scripts/train.py               # end-to-end training on synthetic data
├── src/netflix_fm/
│   ├── config.py                  # typed config dataclasses + YAML loading
│   ├── data/
│   │   ├── schema.py              # Interaction, ItemMetadata, InteractionToken
│   │   ├── tokenizer.py           # interaction tokenization (merge adjacent events)
│   │   ├── features.py            # feature bucketing, ItemCatalog lookup tables
│   │   ├── dataset.py             # sliding-window sampling, multi-step targets
│   │   └── synthetic.py           # synthetic catalog and histories
│   ├── model/
│   │   ├── embeddings.py          # ID + metadata item embedding, age-gated mix
│   │   ├── attention.py           # sparse (local + global) attention, RoPE, KV cache
│   │   ├── transformer.py         # decoder-only backbone
│   │   ├── heads.py               # multi-token item head, auxiliary heads
│   │   └── foundation_model.py    # wires it all together
│   ├── training/
│   │   ├── losses.py              # engagement-weighted multi-token + auxiliary loss
│   │   ├── trainer.py             # training loop, checkpoint save/load
│   │   └── incremental.py         # warm-start retraining, new-title initialisation
│   ├── inference/
│   │   ├── predictor.py           # KV-cached online session: observe / recommend
│   │   └── embeddings.py          # member/item export + Procrustes alignment
│   └── downstream/
│       └── fine_tune.py           # task head on top of the foundation backbone
└── tests/
```

## Blog concept → code

| Concept | Where |
|---|---|
| Interaction tokenization (BPE-like merging of adjacent events, keeping total watch time) | `data/tokenizer.py` |
| Tokens carry action-level (device, duration, time gap, hour) and content-level (item, metadata) info | `data/dataset.py`, `model/embeddings.py` |
| Request-time features (context of the upcoming request) vs action-time features | `data/dataset.py` (`req_*` keys), `inference/predictor.py` |
| Long histories: sparse attention + sliding-window sampling | `model/attention.py`, `data/dataset.py` |
| Efficient inference via KV caching | `model/attention.py` (`KVCache`), `inference/predictor.py` |
| Multi-token prediction (next K items, long-term interest) | `model/heads.py`, `training/losses.py` |
| Auxiliary targets (action type, genre, language) | `model/heads.py`, `training/losses.py` |
| Engagement-weighted loss, short plays masked | `data/dataset.py`, `training/losses.py` |
| Cold start: metadata embeddings mixed with ID embeddings by title age | `model/embeddings.py` |
| Incremental training with new-title initialisation | `training/incremental.py` |
| Embedding stability across retrains | `inference/embeddings.py` |
| Downstream use: embeddings, frozen backbone, fine-tuning | `downstream/fine_tune.py` |

## Quick start

PyTorch wheels need a native arm64 Python on Apple Silicon (an x86_64 conda Python will not find any).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train.py --config configs/default.yaml --out checkpoints/fm.pt
```

## Not yet implemented / next steps

- Real data loaders (the pipeline currently trains on `data/synthetic.py`).
- Sampled softmax for very large catalogs (the item head uses a full softmax).
- Ranking metrics (recall@k, NDCG) in evaluation; only losses are reported now.
- Distributed / mixed-precision training for scaling up model size.
