# foundational-llm-model

Hands-on implementations and notes on **foundation models**: AI agent papers in code, a Netflix-style recommendation foundation model, and a tracker of time-series foundation models.

_Last updated: November 2026_

## Projects

| Folder | What it is | Highlights |
|---|---|---|
| [`ai-agents/`](ai-agents/) | **10 foundational AI agent papers, in code** | Chain-of-Thought, MRKL, ReAct, Toolformer, Reflexion, Self-Refine, Generative Agents, Tree of Thoughts, Voyager and AutoGen, each as a small runnable demo. Runs offline with scripted replies, or live with Claude via `--live`. |
| [`netflix/`](netflix/) | **Netflix-style recommendation foundation model** (PyTorch) | Reference implementation of the Netflix Tech Blog's foundation model for personalized recommendation: interaction tokenization, sparse attention with a KV cache, multi-token prediction, cold-start embeddings, incremental training and downstream fine-tuning. Includes tests and a [blog write-up](netflix/blog/medium-article.md). |
| [Time-series foundation models](#time-series-foundation-models) | **Tracker** (this README) | Comparison of recent forecasting models (TimeH2O-v1, Yug-1, Timestral, TimesFM-3, IBM Granite, Timer-S1, Chronos-2), reported benchmarks and a selection guide. |

### Repository layout

```
foundational-llm-model/
├── ai-agents/            # 01_chain_of_thought.py … 10_autogen.py, common.py (scripted / live LLM)
├── netflix/
│   ├── configs/          # default.yaml hyper-parameters
│   ├── scripts/train.py  # end-to-end training on synthetic data
│   ├── src/netflix_fm/   # data, model, training, inference, downstream
│   ├── tests/            # pytest suite
│   └── blog/             # Medium article + architecture diagram
└── README.md             # this file
```

### Quick start

**AI agents**: no API key needed in the default offline mode.

```bash
cd ai-agents
python3 03_react.py          # offline: scripted LLM replies
python3 03_react.py --live   # real model (needs: pip install anthropic, ANTHROPIC_API_KEY)
```

> In live mode, Reflexion, Voyager and AutoGen execute model-written code on your machine. They are teaching demos, not sandboxes.

**Netflix recommendation model**: trains on synthetic data.

```bash
cd netflix
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train.py --config configs/default.yaml --out checkpoints/fm.pt
```

See each folder's README for details.

---

## Time-series foundation models

Large models pretrained on many datasets that can forecast new series zero-shot, without task-specific training.

- [At a glance](#at-a-glance)
- [Model notes](#model-notes)
- [Reported benchmarks](#reported-benchmarks)
- [Choosing a model](#choosing-a-model)
- [Evaluating on your own data](#evaluating-on-your-own-data)
- [Glossary](#glossary)
- [References](#references)

### At a glance

Newest first. "—" means not confirmed from the sources below.

| Model | Organization | Released | Params | Max context | Multivariate / covariates | License |
|---|---|---|---|---|---|---|
| **TimeH2O-v1** | H2O.ai | Oct 2026 | 108M | 16,384 steps | Covariates | Non-commercial (H2O v1.0) |
| **Yug-1** | Birla AI Labs (India) | Oct 2026 | ~272M | 2,048 points | Covariates | CC BY-NC 4.0 (weights) |
| **Timestral** | Mistral AI | Oct 2026 | 5M–209M (4 sizes) | — | — | Restricts large-company use |
| **TimesFM-3** | Google Research | 2026 | — | — | Multivariate | — |
| **Granite PatchTST-FM-r2 / Timeseries Ensemble r1** | IBM | Mar 2026+ | — | — | — | — |
| **Timer-S1** | Academic | Mar 2026 | ~1B | — | — | — |
| **Chronos-2** | Amazon | Oct 2025 | — | — | Multivariate + covariates | — |

Other common baselines: **Moirai** (Salesforce), **Toto** (Datadog), **TiRex** (NX-AI), **TTM** (IBM).

> ⚠️ Licenses vary widely. Several new releases are **non-commercial**. Check each model card before production use.

### Model notes

- **TimeH2O-v1 (H2O.ai):** Probabilistic forecasts at 9 quantiles (0.1–0.9), an unlimited horizon and optional covariates. H2O.ai reports top scores on GIFT-Eval and fev-bench.
- **Yug-1 (Birla AI Labs):** India-built model aimed at retail demand, finance, equipment monitoring and energy. Supports univariate and covariate-informed forecasting (e.g. promotions, holidays). Available on Hugging Face and GitHub.
- **Timestral (Mistral AI):** Small, CPU-friendly model family. Mistral reports it outperforms TimesFM-3.
- **TimesFM-3 (Google):** Adds zero-shot multivariate forecasting, modelling related series together.
- **IBM Granite time series:** A March 2026 refresh with four models. The Ensemble r1 wraps them in one interface, and IBM is integrating Granite forecasting with Confluent Cloud for streaming data.
- **Timer-S1:** A research model of about 1B parameters that studies scaling for time-series models.
- **Chronos-2 (Amazon):** Moves from univariate to "universal" forecasting with multivariate and covariate support. A strong, widely used baseline.

### Reported benchmarks

All figures are **developer-reported** and not independently reproduced. Lower is better.

| Model | GIFT-Eval MASE | GIFT-Eval CRPS | Source |
|---|---|---|---|
| TimeH2O-v1 | 0.654 | 0.441 | H2O.ai model card |
| IBM TTM-R3-PT | 0.724 | — | Birla AI Labs comparison |
| Salesforce Moirai 2 | 0.728 | — | Birla AI Labs comparison |
| Yug-1 | 0.744 | — | Birla AI Labs comparison |
| TimesFM 2.0 | 0.758 | — | Birla AI Labs comparison |
| Chronos-Bolt Base | 0.808 | — | Birla AI Labs comparison |

TimeH2O-v1 also reports fev-bench skill scores (higher is better) of **50.83** for quantile loss and **39.95** for MASE.

### Choosing a model

| If you need… | Start with |
|---|---|
| Commercial use | Models with permissive licenses. Confirm each license; skip non-commercial ones (TimeH2O-v1, Yug-1) unless you have an agreement. |
| Many related series forecast together | TimesFM-3, Chronos-2 |
| Covariates (promotions, holidays, prices) | Chronos-2, TimeH2O-v1, Yug-1 |
| Long history / long context | TimeH2O-v1 (16k steps) |
| CPU or edge deployment | Timestral (small sizes), IBM Granite |
| Uncertainty / prediction intervals | Quantile-output models such as TimeH2O-v1 |

### Evaluating on your own data

Public leaderboards rarely match your domain, so run a quick backtest before you commit to a model:

1. **Rolling-origin backtest:** forecast from several past cut-off dates, never from a single split.
2. **Simple baselines:** always compare against seasonal naive and a tuned statistical model such as ETS or ARIMA.
3. **Accuracy and calibration:** use MASE for point accuracy, plus CRPS or quantile loss if you need intervals.
4. **Segment the results:** check performance by series type (intermittent, trending, seasonal), not just the overall average.
5. **Cost and latency:** measure inference time on your target hardware.

### Glossary

- **Zero-shot:** forecasting a series the model never saw in training, with no fine-tuning.
- **Covariates:** extra input variables (known future or past) that inform the forecast.
- **MASE:** Mean Absolute Scaled Error. Error relative to a naive forecast, where values below 1 beat naive.
- **CRPS:** Continuous Ranked Probability Score. Measures the quality of a whole probabilistic forecast.
- **GIFT-Eval / fev-bench:** public benchmark suites for time-series foundation models.

### References

- [TimeH2O-v1 – Hugging Face](https://huggingface.co/h2oai/timeh2o-v1)
- [TimeH2O launch – BusinessWire](https://www.financialcontent.com/article/bizwire-2026-10-26-h2oais-timeh2o-beats-mistral-google-and-amazon-in-time-series-forecasting)
- [Yug-1 – Lapaas Voice](https://lapaasvoice.com/birla-ai-labs-launches-yug-1-a-time-series-foundation-model-built-in-india)
- [Timestral – MarkTechPost](https://www.marktechpost.com/2026/10/18/mistral-timestral-time-series-foundation-model/)
- [TimesFM-3 – Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [IBM March 2026 time-series refresh – TSFM.ai](https://tsfm.ai/blog/ibm-march-2026-time-series-refresh)
- [Timer-S1 – arXiv 2603.04791](https://arxiv.org/abs/2603.04791)
- [Chronos-2 – arXiv 2510.15821](https://arxiv.org/abs/2510.15821)
- [Licensing comparison – beri.net](https://www.beri.net/article/timesfm-vs-chronos-vs-moirai-time-series-foundation-model-licence-bigquery-sagemaker-cost-2026)

---

## Contributing

Spotted a new model release, a bug or a correction? Open an issue or a PR. For the tracker, include the model name, release date, a link to the model card or paper, and the license.
