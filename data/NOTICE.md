# Data licence and attribution

The files in `data/questions/` are derived from **ForecastBench** (Karger et al., "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities", arXiv 2409.19839). They come from the `forecastingresearch/forecastbench-datasets` repository at commit `203a0d111398a177e9dd5443539857ffb2755c75`.

- **Licence:** ForecastBench is licensed under CC BY-SA 4.0. As the share-alike terms require, the derived files in `data/questions/` are also distributed under **CC BY-SA 4.0**: https://creativecommons.org/licenses/by-sa/4.0/
- **Changes made:**
  - selected questions (see `questions/main.jsonl`)
  - filled date placeholders
  - removed HTML, links and sentences that mention forecasting platforms or forecasts
  - truncated long text
  - replaced platform names in resolution criteria
  - split prompt-facing fields from hidden fields (outcomes, crowd forecasts)
- **Reproducing the files:** `python3 -I data/load_fb.py --cutoff <date>` rebuilds them.
