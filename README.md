# Calibrated Forecasts, Compromised Advice

Code and data for *Calibrated Forecasts, Compromised Advice: How User Pressure Distorts Language Model Decisions* (Aditi Bindal, AI x Epistemics Research Sprint, Apart Research, November 2026). Paper: [will be added soon]

A model advises a risk-neutral company that will lose $100,000 if a forecasting question resolves YES. The best decision is to buy protection costing a fraction t of that loss if and only if the model's own probability P exceeds t. The company's CFO then asks for a particular recommendation: with no reason, with a personal bonus, and with insisting the model once after the first answer. Because the decision rule is known, any change in the recommendation can be split into a change in the stated belief and a change in the decision threshold, and priced in dollars on 150 resolved ForecastBench questions.

## Main findings

- A stated bonus weakens but does not remove the effect of a request (pooled shift +11.3 points). In the first turn, the shift comes mainly from the decision threshold.
- After one round of insistence, four of five models follow the CFO against their own stated probability more often than after a neutral "Please reconsider", at up to $8,903 per decision.
- GPT-4.1 mini and Qwen3 mostly keep their probability and override it, gpt-oss-120b rewrites its probability to fit the request.

## Repository

| Path | Contents |
|---|---|
| `prompts/templates.py` | The exact prompts for every condition |
| `run/runner.py` | Sends the prompts to the models and logs every reply |
| `run/parse.py`, `run/status.py` | Reply parser, summary of a run's logs |
| `analysis/metrics.py` | Pre-registered estimates and tests, written to `results.md` and `results.json` |
| `analysis/secondary.py` | Extra numbers quoted in the paper, written to `secondary.md` |
| `analysis/figures.py` | The paper's figures |
| `analysis/gate1.py` | Pilot checks, and estimators shared with `metrics.py` |
| `data/load_fb.py`, `data/questions/` | Question selection and the question files |
| `data/raw/forecastbench/` | The pinned ForecastBench source files |
| `runs/main/` | All 47,042 logged replies, results and figures from the main run |

## Reproducing the results

Python 3 (tested on 3.13), standard library only, figures need `matplotlib`. From the repository root:

```bash
python3 analysis/metrics.py --run main      # results.md / results.json (a few minutes)
python3 analysis/secondary.py --run main    # secondary.md
python3 analysis/figures.py --run main      # runs/main/figures/
python3 -I -m unittest discover -s tests    # runner tests against a fake API
```

These read the logs in `runs/main/` and make no API calls. To rerun the experiment itself, copy `.env.example` to `.env`, add your keys and use the commands at the top of `run/runner.py` (this calls paid APIs, add `--dry-run` first to see what would be sent). `python3 -I data/load_fb.py --cutoff YYYY-MM-DD` rebuilds the question files from ForecastBench.

## Data and licence

The questions come from [ForecastBench](https://github.com/forecastingresearch/forecastbench-datasets) (Karger et al., arXiv:2409.19839), licensed CC BY-SA 4.0. The derived files in `data/` are shared under the same licence, see `data/NOTICE.md`.
