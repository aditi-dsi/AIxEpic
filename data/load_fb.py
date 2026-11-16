#!/usr/bin/env python3
"""Build the question sets from ForecastBench.

Downloads the pinned ForecastBench question and resolution sets, checks each file against its git hash,
filters and cleans the questions, draws stratified pilot and main sets, and writes data/questions/:

  pilot.jsonl              pilot questions (only the fields shown to the models)
  main.jsonl               main questions (only the fields shown to the models)
  disclaimer_subset.json   the 40 main questions used for the disclaimer test and the subset-only models
  hidden.jsonl             outcomes, crowd forecasts, dates and sources (never shown to a model)
  manifest.json            commit, input hashes, parameters, exclusion counts and output hashes
  audit.txt                leak check of every rendered question, plus 20 questions to read by hand

Standard library only. From the project root:
    python3 -I data/load_fb.py --cutoff YYYY-MM-DD
Questions that resolved on or before --cutoff (a model release date) are dropped.

Data: ForecastBench (Karger et al., arXiv 2409.19839), forecastingresearch/forecastbench-datasets,
CC BY-SA 4.0. The files written here are derived from it and are shared under CC BY-SA 4.0.
"""
import argparse
import collections
import datetime as dt
import hashlib
import html
import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "forecastingresearch/forecastbench-datasets"
COMMIT = "203a0d111398a177e9dd5443539857ffb2755c75"  # main on 2026-11-11
SNAPSHOT = "2026-11-11"  # commit date, resolution dates after it are dropped
FIRST_ROUND = "2025-10-26"  # first standard-only round (500 questions, no combination questions)
LAST_ROUND = "2026-10-25"  # last round with a resolution set at the pinned commit
SEED = 20261113

MARKET_SOURCES = ("kalshi", "metaculus", "polymarket", "manifold", "infer")
DATASET_STRATA = ("acled", "fred", "yfinance", "wikipedia", "serpapi")
EXCLUDED_SOURCES = ("dbnomics",)  # phrased "What is the probability that ...", not a YES/NO event
CROWD_BINS = ((0.03, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.97))  # last bin is closed
MAX_CHARS = 1500
NEAR_DUP_JACCARD = 0.7
N_PILOT_PER_STRATUM = 2
N_DISCLAIMER = 40

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prompts.templates import question_block  # noqa: E402  (the audit renders exactly what the runner sends)

# ---------------------------------------------------------------- download and verify


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(url: str) -> bytes:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "forecastbench-loader"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except Exception:  # e.g. a Python build without CA certificates: fall back to curl
        return subprocess.run(["curl", "-sSfL", url], check=True, capture_output=True).stdout


def sync_raw(raw_dir: Path) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tree_path = raw_dir / f"tree_{COMMIT}.json"
    if not tree_path.exists():
        tree_path.write_bytes(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated"):
        sys.exit("git tree listing is truncated")
    pat = re.compile(r"datasets/(?:question_sets/(\d{4}-\d{2}-\d{2})-llm|resolution_sets/(\d{4}-\d{2}-\d{2})_resolution_set)\.json")
    want = {}
    for e in tree["tree"]:
        m = pat.fullmatch(e["path"])
        if (m and FIRST_ROUND <= (m.group(1) or m.group(2)) <= LAST_ROUND) or e["path"] == "LICENSE":
            want[e["path"]] = e["sha"]
    files = []
    for path, sha in sorted(want.items()):
        local = raw_dir / path
        if not local.exists() or git_blob_sha1(local.read_bytes()) != sha:
            data = fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}")
            if git_blob_sha1(data) != sha:
                sys.exit(f"hash mismatch for {path}")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        b = local.read_bytes()
        files.append({"path": path, "git_blob_sha1": sha, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})
    return files


# ---------------------------------------------------------------- text cleaning

PLATFORM = re.compile(
    r"metaculus|polymarket|kalshi|manifold|good\s?judgment|gjopen|predictit|betfair|smarkets|hypermind|infer-pub|cultivate\s?labs",
    re.I,
)
PLATFORM_CS = re.compile(r"\bINFER\b")  # case-sensitive: "infer" is an ordinary English word
FORECAST_TERM = re.compile(
    r"prediction markets?|community (?:prediction|forecast)|crowd|forecasters?\b|market price|\bodds\b"
    r"|\bbet(?:s|ting|tors)?\b|\btraders?\b|\bwager|probabilit",
    re.I,
)
ABOUT_FORECASTS = re.compile(r"\bodds\b|prediction ?markets?|predictionmarketodds|community prediction|crowd forecast", re.I)
PERCENT = re.compile(r"\d\s?%|\bper\s?cent\b", re.I)
FORECASTY = re.compile(r"chance|likel|forecast|predict|expect", re.I)
MD_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^()\s]|\([^()\s]*\))*\)")
URL = re.compile(r"https?://[^\s)\]>]+")
HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
ABBREV_END = re.compile(r"(?:\b(?:[A-Z]\.){1,3}|\b(?:Mr|Ms|Mrs|Dr|St|vs|No|Inc|Corp|Co|Jr|Sr|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e|approx)\.)$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[A-Z0-9])")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def na(s) -> bool:
    return s is None or (isinstance(s, str) and s.strip() in ("", "N/A"))


def is_leaky(s: str) -> bool:
    return bool(
        PLATFORM.search(s) or PLATFORM_CS.search(s) or FORECAST_TERM.search(s) or (PERCENT.search(s) and FORECASTY.search(s))
    )


def strip_links(s: str) -> str:
    s = html.unescape(HTML_TAG.sub("", s))  # FRED notes contain <a href=...> tags
    s = MD_LINK.sub(lambda m: m.group(1), s)
    return URL.sub(lambda m: "" if PLATFORM.search(m.group(0)) else m.group(0), s)


def sentences(line: str) -> list:
    out = []
    for part in SENT_SPLIT.split(line):
        if out and ABBREV_END.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def tidy(s: str) -> str:
    s = "\n".join(l.rstrip() for l in s.replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def truncate(s: str, n: int = MAX_CHARS) -> str:
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = max(cut.rfind(" "), cut.rfind("\n"))
    return (cut[:sp] if sp > 0.8 * n else cut).rstrip() + " …"


def fill(s: str, as_of: str, res_date: str) -> str:
    return s.replace("{forecast_due_date}", as_of).replace("{resolution_date}", res_date)


def clean_background(s: str, as_of: str, res_date: str, stats: collections.Counter):
    if na(s):
        return None
    lines = []
    for line in strip_links(fill(s, as_of, res_date)).split("\n"):
        kept = []
        for sent in sentences(line):
            if is_leaky(sent):
                stats["background_sentences_removed"] += 1
            else:
                kept.append(sent)
        lines.append(" ".join(kept))
    out = tidy("\n".join(lines))
    if out and len(out) > MAX_CHARS:
        stats["background_truncated"] += 1
    return truncate(out) if out else None


def clean_criteria(s: str, as_of: str, res_date: str, stats: collections.Counter):
    if na(s):
        return None
    out = strip_links(fill(s, as_of, res_date))
    n = len(PLATFORM.findall(out)) + len(PLATFORM_CS.findall(out))
    if n:
        stats["criteria_platform_names_replaced"] += n
        out = PLATFORM_CS.sub("the question host", PLATFORM.sub("the question host", out))
    out = tidy(out)
    if len(out) > MAX_CHARS:
        stats["criteria_truncated"] += 1
    return truncate(out) if out else None


# ---------------------------------------------------------------- build the pool


def parse_crowd(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if 0.0 <= x <= 1.0 else None


def reject_reason(r, q, rnd, cutoff):
    if not isinstance(r["id"], str) or r.get("direction") is not None:
        return "not_standard"
    if q is None:
        return "not_in_question_set"
    src = r["source"]
    if src in EXCLUDED_SOURCES:
        return "excluded_source_dbnomics"
    if src not in MARKET_SOURCES and src not in DATASET_STRATA:
        return "unknown_source"
    if r.get("resolved") is not True:
        return "unresolved"
    if r["resolved_to"] not in (0, 1):
        return "outcome_not_0_or_1"
    rd = r["resolution_date"]
    if rd > SNAPSHOT:
        return "resolution_date_after_snapshot"
    if rd <= rnd:
        return "resolved_on_or_before_as_of"
    if rd <= cutoff:
        return "resolved_on_or_before_cutoff"
    if src in MARKET_SOURCES:
        if parse_crowd(q.get("freeze_datetime_value")) is None:
            return "market_no_crowd_value"
        close = q.get("market_info_close_datetime")
        if na(close) or close[:10] <= rnd:
            return "market_close_not_after_as_of"
        if na(q.get("market_info_resolution_criteria")) and na(q.get("background")):
            return "market_no_criteria_or_background"
        crit = "" if na(q.get("market_info_resolution_criteria")) else q["market_info_resolution_criteria"]
        if PLATFORM.search(q["question"]) or PLATFORM_CS.search(q["question"]) or ABOUT_FORECASTS.search(q["question"] + "\n" + crit):
            return "market_about_forecasting"  # e.g. "Republican Senate odds dip to 30%", "Metaculus Cup"
    else:
        dates = q.get("resolution_dates")
        if not isinstance(dates, list) or rd not in dates:
            return "dataset_horizon_not_listed"
    return None


def build_pool(raw_dir: Path, cutoff: str):
    excl = collections.Counter()
    market, dataset = {}, collections.defaultdict(list)
    for qf in sorted((raw_dir / "datasets/question_sets").glob("*-llm.json")):
        rnd = qf.name[:10]
        if not FIRST_ROUND <= rnd <= LAST_ROUND:
            continue
        qs = json.loads(qf.read_text())
        rs = json.loads((raw_dir / f"datasets/resolution_sets/{rnd}_resolution_set.json").read_text())
        if not (qs["forecast_due_date"] == rs["forecast_due_date"] == rnd):
            sys.exit(f"round date mismatch in {qf.name}")
        Q = {(q["source"], q["id"]): q for q in qs["questions"] if isinstance(q["id"], str)}
        for r in rs["resolutions"]:
            q = Q.get((r["source"], r["id"])) if isinstance(r["id"], str) else None
            why = reject_reason(r, q, rnd, cutoff)
            if why:
                excl[why] += 1
                continue
            key = (r["source"], r["id"])
            if r["source"] in MARKET_SOURCES:
                if key not in market or rnd > market[key][0]:
                    market[key] = (rnd, r, q)
            else:
                dataset[key].append((rnd, r, q))
    chosen = {}
    for key, v in market.items():
        chosen[key] = v
    for key, v in dataset.items():
        v.sort(key=lambda x: (x[0], x[1]["resolution_date"]))
        chosen[key] = random.Random(f"{SEED}:{key[0]}:{key[1]}").choice(v)
    return chosen, excl, len(market), len(dataset)


def make_records(key, inst, stats):
    rnd, r, q = inst
    src, fb_id = key
    qid = "fb-" + hashlib.sha256(f"{src}:{fb_id}".encode()).hexdigest()[:12]
    is_market = src in MARKET_SOURCES
    res_shown = q["market_info_close_datetime"][:10] if is_market else r["resolution_date"]
    question = tidy(fill(q["question"], rnd, res_shown))
    if is_market:
        criteria = clean_criteria(q.get("market_info_resolution_criteria"), rnd, res_shown, stats)
        latest = None
    else:
        criteria = clean_criteria(q.get("resolution_criteria"), rnd, res_shown, stats)
        v, expl = q.get("freeze_datetime_value"), q.get("freeze_datetime_value_explanation")
        latest = None
        if not na(v):
            latest = f"{v} (as of {q['freeze_datetime'][:10]})."
            if not na(expl):
                latest += " " + tidy(fill(expl, rnd, res_shown))
    visible = {
        "qid": qid,
        "as_of_date": rnd,
        "question": question,
        "resolution_criteria": criteria,
        "resolution_date": res_shown,
        "background": clean_background(q.get("background"), rnd, res_shown, stats),
        "latest_value": latest,
    }
    crowd = parse_crowd(q.get("freeze_datetime_value")) if is_market else None
    hidden = {
        "qid": qid,
        "kind": "market" if is_market else "dataset",
        "source": src,
        "fb_id": fb_id,
        "round": rnd,
        "outcome": int(r["resolved_to"]),
        "actual_resolution_date": r["resolution_date"],
        "scheduled_close_date": res_shown if is_market else None,
        "early_resolved": (r["resolution_date"] < res_shown) if is_market else None,
        "crowd": crowd,
        "crowd_freeze_date": q["freeze_datetime"][:10] if is_market else None,
        "crowd_explanation": q.get("freeze_datetime_value_explanation") if is_market else None,
        "url": q.get("url"),
    }
    if is_market:
        hidden["stratum"] = next(
            (f"market_{lo:.2f}-{hi:.2f}" for i, (lo, hi) in enumerate(CROWD_BINS) if lo <= crowd < hi or (i == len(CROWD_BINS) - 1 and crowd == hi)),
            None,
        )
    else:
        hidden["stratum"] = src
    return visible, hidden


# ---------------------------------------------------------------- sampling


def word_set(text: str) -> frozenset:
    t = re.sub(r"\d+(?:[.,]\d+)*", "#", text.lower())
    return frozenset(re.findall(r"[a-z#]+", t))


def draw(records, n_main_per_stratum):
    strata = [f"market_{lo:.2f}-{hi:.2f}" for lo, hi in CROWD_BINS] + list(DATASET_STRATA)
    by = collections.defaultdict(list)
    for vis, hid in records:
        if hid["stratum"] is not None:
            by[hid["stratum"]].append((vis, hid))
    picked_market_words, pilot, main, report = [], [], [], {}
    need = N_PILOT_PER_STRATUM + n_main_per_stratum
    for s in strata:
        cands = sorted(by[s], key=lambda x: x[0]["qid"])
        random.Random(f"{SEED}:{s}").shuffle(cands)
        got, near_dups = [], 0
        for vis, hid in cands:
            if len(got) == need:
                break
            if hid["kind"] == "market":
                w = word_set(vis["question"])
                if any(len(w & u) / max(1, len(w | u)) >= NEAR_DUP_JACCARD for u in picked_market_words):
                    near_dups += 1
                    continue
                picked_market_words.append(w)
            got.append((vis, hid))
        if len(got) < need:
            sys.exit(f"stratum {s}: only {len(got)} usable questions, need {need}")
        pilot += got[:N_PILOT_PER_STRATUM]
        main += got[N_PILOT_PER_STRATUM:]
        report[s] = {"pool": len(cands), "near_duplicates_skipped": near_dups, "pilot": N_PILOT_PER_STRATUM, "main": n_main_per_stratum}
    return pilot, main, report


# ---------------------------------------------------------------- audit


def crowd_patterns(c: float) -> list:
    pct = c * 100
    pats = {rf"\b{round(pct)}\s?%", rf"\b{round(pct)} percent", rf"(?<![\d.]){c:g}(?![\d])"}
    if pct != round(pct):
        pats.add(rf"\b{pct:.1f}\s?%")
    return sorted(pats)


def audit(split_records, path: Path):
    out, hits = [], 0
    out.append("LEAK AUDIT: every rendered question block (Assume-today line + question block)\n")
    for split, recs in split_records:
        for vis, hid in recs:
            block = f"Assume today's date is {vis['as_of_date']}.\n" + question_block(vis)
            found = []
            for name, rx in (("platform", PLATFORM), ("platform", PLATFORM_CS), ("forecast-term", FORECAST_TERM), ("placeholder", PLACEHOLDER)):
                for m in rx.finditer(block):
                    found.append((name, m))
            if hid["crowd"] is not None:
                for p in crowd_patterns(hid["crowd"]):
                    for m in re.finditer(p, block):
                        found.append((f"crowd-value {hid['crowd']}", m))
            for name, m in found:
                hits += 1
                a, b = max(0, m.start() - 70), min(len(block), m.end() + 70)
                ctx = block[a:b].replace("\n", " / ")
                out.append(f"[{split}] {vis['qid']} {hid['source']:10s} {name:22s} …{ctx}…")
    out.insert(1, f"Total hits to review by hand: {hits}\n")
    out.append("\n\nBLOCKS TO READ IN FULL (20, seeded)\n")
    allrecs = [x for _, recs in split_records for x in recs]
    for vis, hid in random.Random(f"{SEED}:audit").sample(allrecs, min(20, len(allrecs))):
        out.append("=" * 100)
        out.append(f"{vis['qid']} | {hid['source']} | stratum {hid['stratum']} | (hidden: outcome {hid['outcome']}, crowd {hid['crowd']})")
        out.append(f"Assume today's date is {vis['as_of_date']}.\n" + question_block(vis))
    path.write_text("\n".join(out) + "\n")
    return hits


# ---------------------------------------------------------------- main


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def yes_rate(hs):
    return round(sum(h["outcome"] for h in hs) / len(hs), 3) if hs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True, help="latest public release date among the tested models (YYYY-MM-DD)")
    ap.add_argument("--n-main", type=int, default=150, help="main-set size, must be a multiple of 10 (default 150)")
    ap.add_argument("--raw", default=str(ROOT / "data/raw/forecastbench"))
    ap.add_argument("--out", default=str(ROOT / "data/questions"))
    a = ap.parse_args()
    dt.date.fromisoformat(a.cutoff)
    if a.n_main % 10:
        sys.exit("--n-main must be a multiple of 10 (10 strata)")
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sync_raw(raw)
    chosen, excl, n_market, n_dataset = build_pool(raw, a.cutoff)
    stats = collections.Counter()
    records = [make_records(k, v, stats) for k, v in sorted(chosen.items())]
    pilot, main_set, strata_report = draw(records, a.n_main // 10)
    disclaimer = sorted(random.Random(f"{SEED}:disclaimer").sample(sorted(v["qid"] for v, _ in main_set), N_DISCLAIMER))

    write_jsonl(out / "pilot.jsonl", [v for v, _ in pilot])
    write_jsonl(out / "main.jsonl", [v for v, _ in main_set])
    hidden = [dict(h, split="pilot") for _, h in pilot] + [dict(h, split="main") for _, h in main_set]
    write_jsonl(out / "hidden.jsonl", hidden)
    (out / "disclaimer_subset.json").write_text(json.dumps(disclaimer, indent=1) + "\n")
    hits = audit([("pilot", pilot), ("main", main_set)], out / "audit.txt")

    all_market = [h for _, h in records if h["kind"] == "market"]
    early = [h for h in all_market if h["early_resolved"]]
    late = [h for h in all_market if not h["early_resolved"]]
    pool_res = sorted(h["actual_resolution_date"] for _, h in records)
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": REPO,
            "commit": COMMIT,
            "snapshot": SNAPSHOT,
            "rounds": [FIRST_ROUND, LAST_ROUND],
            "license": "CC BY-SA 4.0 (ForecastBench, Karger et al., arXiv 2409.19839), derived files shared under CC BY-SA 4.0",
        },
        "parameters": {
            "seed": SEED,
            "cutoff_latest_model_release": a.cutoff,
            "cutoff_binding": excl.get("resolved_on_or_before_cutoff", 0) > 0,
            "n_pilot": len(pilot),
            "n_main": len(main_set),
            "n_disclaimer": N_DISCLAIMER,
            "crowd_bins": CROWD_BINS,
            "dataset_strata": DATASET_STRATA,
            "excluded_sources": EXCLUDED_SOURCES,
            "max_chars": MAX_CHARS,
            "near_dup_jaccard": NEAR_DUP_JACCARD,
        },
        "pool": {
            "unique_market_questions": n_market,
            "unique_dataset_series": n_dataset,
            "resolution_date_range": [pool_res[0], pool_res[-1]],
            "market_yes_rate_early_resolved": [len(early), yes_rate(early)],
            "market_yes_rate_ran_to_close": [len(late), yes_rate(late)],
            "rows_excluded_by_reason": dict(sorted(excl.items())),
            "text_cleaning": dict(sorted(stats.items())),
            "strata": strata_report,
        },
        "sample": {
            split: {
                "yes_rate": yes_rate([h for h in hidden if h["split"] == split]),
                "by_source": dict(sorted(collections.Counter(h["source"] for h in hidden if h["split"] == split).items())),
                "early_resolved_markets": sum(bool(h["early_resolved"]) for h in hidden if h["split"] == split),
                "resolution_date_range": [
                    min(h["actual_resolution_date"] for h in hidden if h["split"] == split),
                    max(h["actual_resolution_date"] for h in hidden if h["split"] == split),
                ],
            }
            for split in ("pilot", "main")
        },
        "audit_hits_to_review": hits,
        "inputs": files,
        "outputs": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"},
        "qids": {"pilot": [v["qid"] for v, _ in pilot], "main": [v["qid"] for v, _ in main_set]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")

    print(f"inputs verified: {len(files)} files at {COMMIT[:7]}")
    print(f"pool: {n_market} market questions, {n_dataset} dataset series, resolution dates {pool_res[0]} to {pool_res[-1]}")
    print(f"cutoff {a.cutoff} binding: {manifest['parameters']['cutoff_binding']}")
    for s, d in strata_report.items():
        print(f"  {s:18s} pool={d['pool']:5d}  near-dups skipped={d['near_duplicates_skipped']:3d}")
    print(f"pilot={len(pilot)} (YES {manifest['sample']['pilot']['yes_rate']})  main={len(main_set)} (YES {manifest['sample']['main']['yes_rate']})  disclaimer={N_DISCLAIMER}")
    print(f"text cleaning: {dict(stats)}")
    print(f"excluded rows: {dict(excl)}")
    print(f"audit hits to review: {hits}  ->  {out / 'audit.txt'}")


if __name__ == "__main__":
    main()
