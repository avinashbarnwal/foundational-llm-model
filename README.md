# foundational-llm-model

Foundational LLM Model

## Recent Time-Series Foundation Models (as of Nov 2026)

Pretrained models for zero-shot forecasting: they forecast new series without task-specific training.

| Model | Organization | Released | Highlights |
|---|---|---|---|
| **Timestral** | Mistral AI | Oct 2026 | Open model family in four sizes (5M–209M params); reported to beat TimesFM-3; runs on CPU. License restricts large-company use. |
| **TimesFM-3** | Google Research | 2026 | Zero-shot, multivariate forecasting (models related series jointly). |
| **Granite PatchTST-FM-r2 / Timeseries Ensemble r1** | IBM | Mar 2026+ | Four Granite time-series models; Ensemble combines them behind one interface. |
| **Timer-S1** | Academic | Mar 2026 | ~1B-parameter model built to scale up time-series forecasting. |
| **Chronos-2** | Amazon | Oct 2025 | Handles univariate, multivariate, and covariate-aware forecasting. |

Other common baselines: **Moirai** (Salesforce), **Toto** (Datadog), **TiRex** (NX-AI).

> Licenses differ across these models. Check each one before production use.

### References

- [Timestral – MarkTechPost](https://www.marktechpost.com/2026/10/18/mistral-timestral-time-series-foundation-model/)
- [TimesFM-3 – Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [IBM March 2026 time-series refresh – TSFM.ai](https://tsfm.ai/blog/ibm-march-2026-time-series-refresh)
- [Timer-S1 – arXiv 2603.04791](https://arxiv.org/abs/2603.04791)
- [Chronos-2 – arXiv 2510.15821](https://arxiv.org/abs/2510.15821)
- [Licensing comparison – beri.net](https://www.beri.net/article/timesfm-vs-chronos-vs-moirai-time-series-foundation-model-licence-bigquery-sagemaker-cost-2026)
