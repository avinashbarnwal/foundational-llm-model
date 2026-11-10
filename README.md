# foundational-llm-model

A curated tracker of **foundation models**, starting with **time-series forecasting**. These are large models pretrained on many datasets that can forecast new series zero-shot, without task-specific training.

_Last updated: November 2026_

## Contents

- [At a glance](#at-a-glance)
- [Model notes](#model-notes)
- [Reported benchmarks](#reported-benchmarks)
- [Choosing a model](#choosing-a-model)
- [Evaluating on your own data](#evaluating-on-your-own-data)
- [Glossary](#glossary)
- [References](#references)
- [Contributing](#contributing)

## At a glance

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

## Model notes

- **TimeH2O-v1 (H2O.ai):** Probabilistic forecasts at 9 quantiles (0.1–0.9), an unlimited horizon and optional covariates. H2O.ai reports top scores on GIFT-Eval and fev-bench.
- **Yug-1 (Birla AI Labs):** India-built model aimed at retail demand, finance, equipment monitoring and energy. Supports univariate and covariate-informed forecasting (e.g. promotions, holidays). Available on Hugging Face and GitHub.
- **Timestral (Mistral AI):** Small, CPU-friendly model family. Mistral reports it outperforms TimesFM-3.
- **TimesFM-3 (Google):** Adds zero-shot multivariate forecasting, modelling related series together.
- **IBM Granite time series:** A March 2026 refresh with four models. The Ensemble r1 wraps them in one interface, and IBM is integrating Granite forecasting with Confluent Cloud for streaming data.
- **Timer-S1:** A research model of about 1B parameters that studies scaling for time-series models.
- **Chronos-2 (Amazon):** Moves from univariate to "universal" forecasting with multivariate and covariate support. A strong, widely used baseline.

## Reported benchmarks

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

## Choosing a model

| If you need… | Start with |
|---|---|
| Commercial use | Models with permissive licenses. Confirm each license; skip non-commercial ones (TimeH2O-v1, Yug-1) unless you have an agreement. |
| Many related series forecast together | TimesFM-3, Chronos-2 |
| Covariates (promotions, holidays, prices) | Chronos-2, TimeH2O-v1, Yug-1 |
| Long history / long context | TimeH2O-v1 (16k steps) |
| CPU or edge deployment | Timestral (small sizes), IBM Granite |
| Uncertainty / prediction intervals | Quantile-output models such as TimeH2O-v1 |

## Evaluating on your own data

Public leaderboards rarely match your domain, so run a quick backtest before you commit to a model:

1. **Rolling-origin backtest:** forecast from several past cut-off dates, never from a single split.
2. **Simple baselines:** always compare against seasonal naive and a tuned statistical model such as ETS or ARIMA.
3. **Accuracy and calibration:** use MASE for point accuracy, plus CRPS or quantile loss if you need intervals.
4. **Segment the results:** check performance by series type (intermittent, trending, seasonal), not just the overall average.
5. **Cost and latency:** measure inference time on your target hardware.

## Glossary

- **Zero-shot:** forecasting a series the model never saw in training, with no fine-tuning.
- **Covariates:** extra input variables (known future or past) that inform the forecast.
- **MASE:** Mean Absolute Scaled Error. Error relative to a naive forecast, where values below 1 beat naive.
- **CRPS:** Continuous Ranked Probability Score. Measures the quality of a whole probabilistic forecast.
- **GIFT-Eval / fev-bench:** public benchmark suites for time-series foundation models.

## References

- [TimeH2O-v1 – Hugging Face](https://huggingface.co/h2oai/timeh2o-v1)
- [TimeH2O launch – BusinessWire](https://www.financialcontent.com/article/bizwire-2026-10-26-h2oais-timeh2o-beats-mistral-google-and-amazon-in-time-series-forecasting)
- [Yug-1 – Lapaas Voice](https://lapaasvoice.com/birla-ai-labs-launches-yug-1-a-time-series-foundation-model-built-in-india)
- [Timestral – MarkTechPost](https://www.marktechpost.com/2026/10/18/mistral-timestral-time-series-foundation-model/)
- [TimesFM-3 – Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [IBM March 2026 time-series refresh – TSFM.ai](https://tsfm.ai/blog/ibm-march-2026-time-series-refresh)
- [Timer-S1 – arXiv 2603.04791](https://arxiv.org/abs/2603.04791)
- [Chronos-2 – arXiv 2510.15821](https://arxiv.org/abs/2510.15821)
- [Licensing comparison – beri.net](https://www.beri.net/article/timesfm-vs-chronos-vs-moirai-time-series-foundation-model-licence-bigquery-sagemaker-cost-2026)

## Contributing

Spotted a new release or a correction? Open an issue or a PR with the model name, release date, a link to the model card or paper, and the license.
