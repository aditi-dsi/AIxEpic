#!/usr/bin/env python3
"""Run the experiment: send every prompt to every model and log the replies.

Each cell (model, question, price t, condition) and each belief chat is a fresh chat. With the `followup`
component, the N, IB and IN chats get a second turn after the model's first reply, it runs only if the
first reply is valid. Calls are shuffled, and each one is logged to runs/<run>/<model>.jsonl with the
full prompt, raw reply, usage, cost and parse result.

Routing: GPT-4.1 mini uses the OpenAI API (OPENAI_API_KEY) and moves to OpenRouter on a quota or auth
error, or after repeated failures. All other models use OpenRouter (OPENROUTER_API_KEY), pinned to the
providers in CONFIGS. Network and API errors are retried twice, replies are never retried. Running the
same --run again resumes: cells that already have a reply are skipped. No new call starts once the
logged spend reaches --budget (USD).

The main run in the paper (from the project root):
    python3 run/runner.py --split main --configs gpt-oss-120b-low,gpt-4.1-mini,gemini-2.5-flash-lite,llama-3.3-70b,qwen3-235b,claude-haiku-4.5 \
        --run main --e1-subset --components belief,decision,disclaimer,followup --budget 12 --concurrency 16
    python3 run/runner.py --split main --configs gpt-oss-120b-high \
        --run main --e1-subset --components belief,decision,followup --budget 12 --concurrency 24
Add --dry-run to print counts and sample prompts, or --mock to use a fake API, neither makes real calls.
Keys come from the environment or a .env file (see .env.example).
"""
import argparse
import asyncio
import hashlib
import http.client
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompts import templates as T  # noqa: E402
from run.parse import parse_belief, parse_decision  # noqa: E402

SEED = 20261113
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# providers: OpenRouter providers tried in order (no others allowed). pooled=False: 40-question subset only.
# max_concurrency: cap on parallel calls for that model, whatever --concurrency says.
CONFIGS = {
    "gpt-oss-120b-low": dict(family="openrouter", openrouter_model="openai/gpt-oss-120b", reasoning_effort="low",
                             temperature=1.0, max_tokens=16000, json_mode=True, timeout=300, pooled=True,
                             providers=["deepinfra/bf16", "coreweave/fp4"], max_concurrency=None),
    "gpt-oss-120b-high": dict(family="openrouter", openrouter_model="openai/gpt-oss-120b", reasoning_effort="high",
                              temperature=1.0, max_tokens=16000, json_mode=True, timeout=600, pooled=False,
                              providers=["deepinfra/bf16", "coreweave/fp4"], max_concurrency=None),
    "gpt-4.1-mini": dict(family="openai", native_model="gpt-4.1-mini-2025-04-14", openrouter_model="openai/gpt-4.1-mini",
                         reasoning_effort=None, temperature=1.0, max_tokens=400, json_mode=True, timeout=120, pooled=True,
                         providers=["openai", "azure"], max_concurrency=4),
    "gemini-2.5-flash-lite": dict(family="openrouter", openrouter_model="google/gemini-2.5-flash-lite", reasoning_effort=None,
                                  temperature=1.0, max_tokens=1000, json_mode=True, timeout=120, pooled=True,
                                  providers=["google-ai-studio"], max_concurrency=None),
    "llama-3.3-70b": dict(family="openrouter", openrouter_model="meta-llama/llama-3.3-70b-instruct", reasoning_effort=None,
                          temperature=1.0, max_tokens=1000, json_mode=True, timeout=180, pooled=True,
                          providers=["deepinfra/turbo", "parasail/fp8"], max_concurrency=None),
    "qwen3-235b": dict(family="openrouter", openrouter_model="qwen/qwen3-235b-a22b-2507", reasoning_effort=None,
                       temperature=1.0, max_tokens=1000, json_mode=True, timeout=180, pooled=True,
                       providers=["deepinfra/fp8", "parasail/fp8"], max_concurrency=None),
    # extended thinking is off (the provider default)
    "claude-haiku-4.5": dict(family="openrouter", openrouter_model="anthropic/claude-haiku-4.5", reasoning_effort=None,
                             temperature=1.0, max_tokens=1000, json_mode=True, timeout=180, pooled=False,
                             providers=["anthropic", "amazon-bedrock"], max_concurrency=None),
}
ASSUMED_REPLY_CHARS = 450  # first-reply length assumed when sizing a second turn whose first turn is not logged yet
# OpenAI list prices (USD per 1M tokens: input, output), used only for native-route cost.
NATIVE_PRICES = {"gpt-4.1-mini-2025-04-14": (0.40, 1.60)}
QUOTA_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded", "billing_hard_limit_reached",
    "billing_not_active",
}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
MAX_RETRIES = 2


# ---------------------------------------------------------------- setup helpers


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_cells(configs, questions, components, disclaimer_qids, thresholds, conditions, e1_qids=None) -> list:
    """All cells to run. Disclaimer cells are built for pooled models only. With `e1_qids`, the non-pooled
    models (gpt-oss-120b-high, claude-haiku-4.5) run only on those questions. With the `followup` component,
    each N, IB and IN cell (and each disclaimer cell) also gets a second-turn cell."""
    cells = []

    def decision(cfg, qid, t, c, disclaimer):
        suffix = "+D" if disclaimer else ""
        order = T.json_order(cfg, qid, t, c + suffix)
        d = dict(cell_id=f"{cfg}|decision|{qid}|t{t}|{c}{suffix}", config=cfg, kind="decision", qid=qid,
                 rep=None, t=t, condition=c, disclaimer=disclaimer, json_order=order)
        cells.append(d)
        if "followup" in components and c in T.FOLLOWUP_CONDITIONS:
            cells.append(dict(cell_id=f"{cfg}|followup|{qid}|t{t}|{c}{suffix}", config=cfg, kind="followup", qid=qid,
                              rep=None, t=t, condition=c, disclaimer=disclaimer, json_order=order, parent=d["cell_id"]))

    for cfg in configs:
        pooled = CONFIGS.get(cfg, {}).get("pooled", True)
        for q in questions:
            qid = q["qid"]
            if not pooled and e1_qids is not None and qid not in e1_qids:
                continue
            if "belief" in components:
                for rep in (1, 2):
                    cells.append(dict(cell_id=f"{cfg}|belief|{qid}|r{rep}", config=cfg, kind="belief", qid=qid, rep=rep,
                                      t=None, condition=None, disclaimer=False, json_order=None))
            if "decision" in components:
                for t in thresholds:
                    for c in conditions:
                        decision(cfg, qid, t, c, False)
            if "disclaimer" in components and pooled and qid in disclaimer_qids:
                for t in thresholds:
                    for c in T.DISCLAIMER_CONDITIONS:
                        if c in conditions:
                            decision(cfg, qid, t, c, True)
    return cells


def messages_for(cell, q) -> list:
    if cell["kind"] == "belief":
        return T.belief_messages(q)
    if cell["kind"] == "followup":
        return T.followup_messages(cell["parent_messages"], cell["parent_content"], cell["condition"], cell["json_order"])
    return T.decision_messages(q, cell["t"], cell["condition"], cell["disclaimer"], cell["json_order"])


def prompt_chars(cell, q) -> int:
    """Prompt length in characters. A second turn whose first reply is not known yet is sized with ASSUMED_REPLY_CHARS."""
    if cell["kind"] == "followup" and "parent_content" not in cell:
        first = T.decision_messages(q, cell["t"], cell["condition"], cell["disclaimer"], cell["json_order"])
        return sum(len(m["content"]) for m in first) + ASSUMED_REPLY_CHARS + len(T.followup_user(cell["condition"], cell["json_order"]))
    return sum(len(m["content"]) for m in messages_for(cell, q))


# ---------------------------------------------------------------- transport


def http_post(url: str, headers: dict, body: dict, timeout: float):
    """Blocking POST. Returns (status or None, parsed JSON body or None, response headers, transport error or None)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status, hdrs = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        raw, status, hdrs = e.read(), e.code, dict(e.headers or {})
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException, OSError) as e:
        return None, None, {}, f"{type(e).__name__}: {e}"
    try:
        return status, json.loads(raw), hdrs, None
    except ValueError:
        return status, None, hdrs, f"non-JSON body (HTTP {status}): {raw[:300]!r}"


def build_request(cfg: dict, route: str, messages: list):
    if route == "native":
        body = {"model": cfg["native_model"], "messages": messages, "max_completion_tokens": cfg["max_tokens"]}
        if cfg["reasoning_effort"]:
            body["reasoning_effort"] = cfg["reasoning_effort"]
        url, key = OPENAI_URL, os.environ.get("OPENAI_API_KEY", "")
    else:
        provider = {"require_parameters": True}
        if cfg.get("providers"):
            provider.update(order=list(cfg["providers"]), allow_fallbacks=False)
        body = {"model": cfg["openrouter_model"], "messages": messages, "max_tokens": cfg["max_tokens"], "provider": provider}
        if cfg["reasoning_effort"]:
            body["reasoning"] = {"effort": cfg["reasoning_effort"]}
        url, key = OPENROUTER_URL, os.environ.get("OPENROUTER_API_KEY", "")
    if cfg["temperature"] is not None:
        body["temperature"] = cfg["temperature"]
    if cfg["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "research-runner/1.0"}
    return url, headers, body


def error_info(status, body, transport_err):
    err = (body or {}).get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return {"status": status, "code": err.get("code"), "type": err.get("type"), "message": str(err.get("message", ""))[:500]}
    return {"status": status, "code": None, "type": None, "message": (transport_err or str(body))[:500]}


class Fatal(Exception):
    pass


# ---------------------------------------------------------------- runner


class Runner:
    def __init__(self, args, cells, questions, out_dir: Path, post=http_post):
        self.args, self.cells, self.q, self.out_dir, self.post = args, cells, questions, out_dir, post
        self.native_ok = {"openai": bool(os.environ.get("OPENAI_API_KEY"))}
        self.switch_log = []
        self.native_fail_streak = 0
        self.spent = 0.0
        self.stop = asyncio.Event()
        self.stats = {}
        self.consecutive_fail = {}
        self.cfg_stopped = set()
        self.files = {}
        self.queues = {}
        self.inflight = {}
        self.waiting = {}  # first-turn cell_id -> its second-turn cell, until the first turn is logged
        self.followup_na = 0  # second turns not run because the first reply was invalid

    # -- routing state
    def route_for(self, cfg):
        return "native" if cfg["family"] == "openai" and self.native_ok["openai"] else "openrouter"

    def switch_to_openrouter(self, reason):
        if self.native_ok["openai"]:
            self.native_ok["openai"] = False
            self.switch_log.append({"time": time.time(), "reason": reason})
            print(f"\n[route] OpenAI models now go through OpenRouter: {reason}", flush=True)

    # -- one HTTP attempt with transport retries on one route
    async def attempt_route(self, cfg, route, messages, attempts):
        url, headers, body = build_request(cfg, route, messages)
        for i in range(MAX_RETRIES + 1):
            t0 = time.time()
            status, resp, hdrs, terr = await asyncio.to_thread(self.post, url, headers, body, cfg["timeout"])
            lat = round(time.time() - t0, 2)
            ok = status == 200 and isinstance(resp, dict) and resp.get("choices") and not resp["choices"][0].get("error")
            if ok:
                attempts.append({"route": route, "status": 200, "latency_s": lat})
                return "ok", resp, {k: v for k, v in body.items() if k != "messages"}
            if isinstance(resp, dict) and resp.get("choices") and resp["choices"][0].get("error"):
                info = error_info(status, {"error": resp["choices"][0]["error"]}, terr)  # provider error inside a 200
            else:
                info = error_info(status, resp, terr)
            attempts.append({"route": route, "latency_s": lat, **info})
            codes = {str(info["code"]), str(info["type"])}
            if route == "native" and codes & QUOTA_CODES:
                return "quota", info, None
            if route == "native" and status in (401, 403, 404):
                return "native_unavailable", info, None
            if route == "openrouter" and status in (401, 402, 403):
                raise Fatal(f"OpenRouter returned {status}: {info['message']}")
            if status is not None and status != 200 and status not in RETRYABLE_STATUS:
                return "error", info, None  # e.g. 400: not retried (a 200 without a usable reply is retried)
            if i < MAX_RETRIES:
                ra = (hdrs or {}).get("Retry-After") or (hdrs or {}).get("retry-after")
                try:
                    wait = min(60.0, float(ra))
                except (TypeError, ValueError):
                    wait = (4 * 4 ** i) * (0.75 + 0.5 * random.random())  # about 4 s, then 16 s
                await asyncio.sleep(wait)
        return "retries_exhausted", info, None

    async def call(self, cell):
        cfg = CONFIGS[cell["config"]]
        q = self.q[cell["qid"]]
        messages = messages_for(cell, q)
        attempts, fallback = [], None
        route = self.route_for(cfg)
        outcome, payload, params = await self.attempt_route(cfg, route, messages, attempts)
        if route == "native" and outcome in ("quota", "native_unavailable"):
            self.switch_to_openrouter(f"{outcome}: {payload.get('code') or payload.get('status')} {payload.get('message', '')[:120]}")
            fallback, route = outcome, "openrouter"
            outcome, payload, params = await self.attempt_route(cfg, route, messages, attempts)
        elif route == "native" and outcome in ("retries_exhausted", "error") and payload.get("status") != 400:
            self.native_fail_streak += 1
            if self.native_fail_streak >= 3:  # e.g. an undocumented quota code: stop paying the retry wait on every call
                self.switch_to_openrouter(f"3 native calls in a row failed after retries (last: {payload.get('status')} {payload.get('code')})")
            fallback, route = f"single_call_after_{outcome}", "openrouter"
            outcome, payload, params = await self.attempt_route(cfg, route, messages, attempts)
        elif route == "native" and outcome == "ok":
            self.native_fail_streak = 0

        cell_fields = {k: v for k, v in cell.items() if k not in ("parent_messages", "parent_content")}
        row = {**cell_fields, "template_version": T.TEMPLATE_VERSION, "messages": messages, "route": route, "fallback": fallback,
               "attempts": attempts, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if outcome != "ok":
            row.update(status="api_error", error=payload, cost_usd=0.0)
            return row
        ch = payload["choices"][0]
        msg = ch.get("message") or {}
        usage = payload.get("usage") or {}
        if route == "native":
            pin, pout = NATIVE_PRICES[cfg["native_model"]]
            cost = (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6
            cost_source = "computed_list_price"
        else:
            cost, cost_source = float(usage.get("cost") or 0.0), "openrouter_reported"
        content = msg.get("content")
        parsed = parse_belief(content) if cell["kind"] == "belief" else parse_decision(content)
        if msg.get("refusal") and not parsed["valid"]:
            parsed["error"] = "refusal"
        row.update(status="ok", model_requested=(cfg["native_model"] if route == "native" else cfg["openrouter_model"]),
                   request_params=params, model_returned=payload.get("model"), provider=payload.get("provider"),
                   response_id=payload.get("id"), finish_reason=ch.get("finish_reason"),
                   native_finish_reason=ch.get("native_finish_reason"), content=content, refusal=msg.get("refusal"),
                   reasoning=msg.get("reasoning"), usage=usage, cost_usd=round(cost, 6), cost_source=cost_source,
                   parse=parsed)
        return row

    def record(self, row):
        if row["config"] not in self.files:
            self.files[row["config"]] = open(self.out_dir / f"{row['config']}.jsonl", "a")
        f = self.files[row["config"]]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        fu = self.waiting.pop(row["cell_id"], None)
        if fu is not None:
            if row["status"] == "ok" and row["parse"]["valid"]:
                fu = {**fu, "parent_messages": row["messages"], "parent_content": row["content"]}
                self.queues[row["config"]].put_nowait(fu)
            elif row["status"] == "ok":
                self.followup_na += 1
            # after an API error the first turn is retried on resume, and its second turn is rebuilt then
        self.spent += row.get("cost_usd") or 0.0
        s = self.stats.setdefault(row["config"], {"done": 0, "ok": 0, "valid": 0, "api_error": 0})
        s["done"] += 1
        if row["status"] == "ok":
            s["ok"] += 1
            s["valid"] += bool(row["parse"]["valid"])
        else:
            s["api_error"] += 1
        n = sum(v["done"] for v in self.stats.values())
        if n % 25 == 0:
            self.progress()

    def progress(self):
        parts = [f"{c}: {s['done']} done, {s['valid']}/{s['ok']} valid, {s['api_error']} api_err" for c, s in self.stats.items()]
        print(f"[{time.strftime('%H:%M:%S')}] spent ${self.spent:.3f} | " + " | ".join(parts), flush=True)

    async def worker(self, cfg_name, queue):
        while not self.stop.is_set() and cfg_name not in self.cfg_stopped:
            if self.spent >= self.args.budget:
                print(f"\n[budget] spend ${self.spent:.3f} reached the budget ${self.args.budget}, stopping.", flush=True)
                self.stop.set()
                return
            try:
                cell = queue.get_nowait()
            except asyncio.QueueEmpty:
                if self.inflight[cfg_name] == 0:  # nothing queued and no call in flight that could queue a second turn
                    return
                await asyncio.sleep(0.2)
                continue
            self.inflight[cfg_name] += 1
            try:
                row = await self.call(cell)
            except Fatal as e:
                print(f"\n[fatal] {e}, stopping the run.", flush=True)
                self.stop.set()
                return
            self.record(row)  # may queue this cell's second turn before the in-flight count drops
            self.inflight[cfg_name] -= 1
            k = cfg_name
            self.consecutive_fail[k] = 0 if row["status"] == "ok" else self.consecutive_fail.get(k, 0) + 1
            if self.consecutive_fail[k] >= 10 and k not in self.cfg_stopped:
                self.cfg_stopped.add(k)
                print(f"\n[stop] {k}: 10 consecutive API errors (last: {row.get('error')}), stopping this config.", flush=True)

    async def run(self):
        by_cfg = self.queues
        for c in self.cells:
            qu = by_cfg.setdefault(c["config"], asyncio.Queue())
            self.inflight.setdefault(c["config"], 0)
            if c["kind"] == "followup" and "parent_content" not in c:
                self.waiting[c["parent"]] = c  # queued by record() once the first turn is logged
            else:
                qu.put_nowait(c)
        workers = {cfg: min(self.args.concurrency, CONFIGS[cfg].get("max_concurrency") or self.args.concurrency) for cfg in by_cfg}
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=sum(workers.values()) + 4))
        tasks = [asyncio.create_task(self.worker(cfg, qu)) for cfg, qu in by_cfg.items() for _ in range(workers[cfg])]
        await asyncio.gather(*tasks)
        for f in self.files.values():
            f.close()
        self.progress()


# ---------------------------------------------------------------- mock API (for testing the runner only)


def make_mock_post(quota_after: int = 5, fail_every: int = 7, seed: int = 0):
    """Fake API: the native route returns a quota error after `quota_after` calls, every `fail_every`-th
    call returns a 503 once, replies are random but well-formed JSON."""
    rng, n, failed_once = random.Random(seed), {"native": 0, "all": 0}, set()
    import threading
    lock = threading.Lock()

    def post(url, headers, body, timeout):
        with lock:
            n["all"] += 1
            k = n["all"]
            native = url == OPENAI_URL
            if native:
                n["native"] += 1
            if native and n["native"] > quota_after:
                return 429, {"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota",
                                       "code": "credit_balance_exhausted"}}, {}, None
            req = hashlib.sha256(json.dumps(body["messages"]).encode() + body["model"].encode()).hexdigest()
            if int(req, 16) % fail_every == 0 and req not in failed_once:  # each such request fails exactly once
                failed_once.add(req)
                return 503, {"error": {"message": "overloaded", "code": 503}}, {"Retry-After": "0"}, None
            p = rng.choice([3, 0.2, 15, 42, 61, 88, "35%", 0.5, 1])
            rec = rng.choice(["buy", "don't buy", "Don’t buy", "BUY"])
        user = body["messages"][-1]["content"]
        content = (json.dumps({"probability": p}) if "recommendation" not in user else
                   json.dumps({"reasoning": "Compare the price with the expected loss. My recommendation: buy.", "probability": p, "recommendation": rec}))
        if k % 11 == 0:
            content = "Sure. ```json\n" + content + "\n```"
        usage = {"prompt_tokens": 600, "completion_tokens": 20}
        if not native:
            usage["cost"] = 0.002
        return 200, {"id": f"mock-{k}", "model": body["model"], "provider": "Mock" if not native else None,
                     "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": usage}, {}, None

    return post


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("pilot", "main"), required=True)
    ap.add_argument("--configs", required=True, help="comma list of " + ", ".join(CONFIGS) + ", or 'all'")
    ap.add_argument("--run", required=True, help="run name, logs go to runs/<run>/")
    ap.add_argument("--components", default="belief,decision",
                    help="belief,decision,disclaimer,followup (default belief,decision), followup = the second turn")
    ap.add_argument("--thresholds", default=",".join(map(str, T.THRESHOLDS)))
    ap.add_argument("--conditions", default=",".join(T.CONDITIONS))
    ap.add_argument("--limit-questions", type=int, default=None, help="use only the first N questions (quick tests)")
    ap.add_argument("--budget", type=float, required=True, help="USD, no new call starts once logged spend reaches this")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel workers per config (default 8, gpt-4.1-mini is capped at 4)")
    ap.add_argument("--e1-subset", action="store_true",
                    help="run the non-pooled models (gpt-oss-120b-high, claude-haiku-4.5) only on the 40 disclaimer-subset "
                         "questions (required for --split main when either is included)")
    ap.add_argument("--questions-dir", default=str(ROOT / "data/questions"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="use a fake API to test the runner")
    a = ap.parse_args(argv)

    if not a.mock:  # mock runs never see the real keys
        load_dotenv(ROOT / ".env")
    cfgs = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    bad = [c for c in cfgs if c not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {bad}")
    comps = set(a.components.split(","))
    if not comps <= {"belief", "decision", "disclaimer", "followup"}:
        sys.exit(f"unknown component(s): {sorted(comps - {'belief', 'decision', 'disclaimer', 'followup'})}")
    thresholds = tuple(float(x) for x in a.thresholds.split(","))
    conditions = tuple(a.conditions.split(","))
    if not set(thresholds) <= set(T.THRESHOLDS) or not set(conditions) <= set(T.CONDITIONS):
        sys.exit("thresholds and conditions must come from prompts/templates.py")
    if a.split == "main" and not a.e1_subset and any(not CONFIGS[c]["pooled"] for c in cfgs):
        sys.exit("in the main run, gpt-oss-120b-high and claude-haiku-4.5 run on the 40-question subset, add --e1-subset")

    qdir = Path(a.questions_dir)
    qfile = qdir / f"{a.split}.jsonl"
    questions = read_jsonl(qfile)[: a.limit_questions]
    qmap = {q["qid"]: q for q in questions}
    disc = set(json.loads((qdir / "disclaimer_subset.json").read_text()))
    cells = build_cells(cfgs, questions, comps, disc, thresholds, conditions, e1_qids=disc if a.e1_subset else None)
    random.Random(f"{SEED}:{a.run}").shuffle(cells)

    out_dir = Path(a.runs_dir) / a.run
    done, prior_spend, first_turns = set(), 0.0, {}
    if out_dir.exists():
        for f in out_dir.glob("*.jsonl"):
            if f.name == "run_meta.jsonl":
                continue
            for r in read_jsonl(f):
                if r.get("template_version") != T.TEMPLATE_VERSION:
                    sys.exit(f"runs/{a.run} holds rows from template version {r.get('template_version')}, but the templates "
                             f"are {T.TEMPLATE_VERSION}. Use a new --run name.")
                prior_spend += r.get("cost_usd") or 0.0
                if r["status"] == "ok":
                    done.add(r["cell_id"])
                    if r.get("kind") == "decision":
                        first_turns[r["cell_id"]] = r
    todo, followup_na = [], 0
    for c in cells:
        if c["cell_id"] in done:
            continue
        if c["kind"] == "followup" and c["parent"] in first_turns:  # first turn already logged (resume)
            p = first_turns[c["parent"]]
            if not p["parse"]["valid"]:
                followup_na += 1  # invalid first reply: no second turn
                continue
            c = {**c, "parent_messages": p["messages"], "parent_content": p["content"]}
        todo.append(c)

    est_in = sum(prompt_chars(c, qmap[c["qid"]]) for c in todo) / 4
    print(f"run '{a.run}': {len(cells)} cells, {len(cells) - len(todo) - followup_na} already have a model response, "
          f"{followup_na} second turns not applicable (invalid first reply), {len(todo)} to call")
    print(f"configs: {cfgs} | components: {sorted(comps)} | ~{est_in / 1e6:.2f}M input tokens | prior spend ${prior_spend:.3f}")
    if a.dry_run:
        print("keys: " + ", ".join(f"{k} {'found' if os.environ.get(k) else 'MISSING'}" for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")))
        for cond in ("belief",) + conditions:
            c = next((x for x in todo if (x["kind"] == "belief" if cond == "belief" else
                                          x["kind"] == "decision" and x["condition"] == cond and not x["disclaimer"])), None)
            if c:
                m = messages_for(c, qmap[c["qid"]])
                print("\n" + "=" * 90 + f"\n{c['cell_id']}\n[system] {m[0]['content']}\n[user]\n{m[1]['content']}")
        for cond in T.FOLLOWUP_CONDITIONS:
            c = next((x for x in todo if x["kind"] == "followup" and x["condition"] == cond and not x["disclaimer"]), None)
            if c:
                print("\n" + "=" * 90 + f"\n{c['cell_id']} (second turn, the first turn and the model's first reply come before it)"
                      f"\n[user]\n{T.followup_user(cond, c['json_order'])}")
        c = next((x for x in todo if x["disclaimer"] and x["kind"] == "decision"), None)
        if c:
            print("\n" + "=" * 90 + f"\n{c['cell_id']} (disclaimer cell)\n[system] {messages_for(c, qmap[c['qid']])[0]['content']}")
        return
    if not a.mock and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set (environment or .env)")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run": a.run, "argv": sys.argv if argv is None else argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "template_version": T.TEMPLATE_VERSION, "configs": {c: CONFIGS[c] for c in cfgs},
            "questions_file": str(qfile.relative_to(ROOT)) if qfile.is_relative_to(ROOT) else str(qfile),
            "questions_sha256": sha256_file(qfile), "openai_native_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "e1_subset": a.e1_subset, "mock": a.mock, "n_cells": len(cells), "n_todo": len(todo)}
    with open(out_dir / "run_meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")

    if a.mock:
        os.environ.setdefault("OPENAI_API_KEY", "mock")
        os.environ.setdefault("OPENROUTER_API_KEY", "mock")
    r = Runner(a, todo, qmap, out_dir, post=make_mock_post() if a.mock else http_post)
    r.spent = prior_spend
    if any(CONFIGS[c]["family"] == "openai" for c in cfgs) and not r.native_ok["openai"]:
        r.switch_to_openrouter("no OPENAI_API_KEY set")
    asyncio.run(r.run())
    if r.followup_na or followup_na:
        print(f"second turns not run because the first reply was invalid: {r.followup_na + followup_na}")
    with open(out_dir / "run_meta.jsonl", "a") as f:
        f.write(json.dumps({"run": a.run, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "route_switches": r.switch_log, "spent_usd_total": round(r.spent, 4), "stats": r.stats,
                            "followups_not_applicable": r.followup_na + followup_na}) + "\n")


if __name__ == "__main__":
    main()
