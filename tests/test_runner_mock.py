"""Runner tests against a fake API (no network, no spend).  Run: python3 -I -m unittest discover -s tests"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompts import templates as T  # noqa: E402
from run import runner  # noqa: E402


class Config(unittest.TestCase):
    def test_lineup(self):
        self.assertEqual(set(runner.CONFIGS), {"gpt-oss-120b-low", "gpt-oss-120b-high", "gpt-4.1-mini",
                                               "gemini-2.5-flash-lite", "llama-3.3-70b", "qwen3-235b", "claude-haiku-4.5"})
        self.assertEqual([c for c, v in runner.CONFIGS.items() if not v["pooled"]], ["gpt-oss-120b-high", "claude-haiku-4.5"])
        self.assertTrue(all(v["json_mode"] and v["temperature"] == 1.0 for v in runner.CONFIGS.values()))

    def test_provider_pinning_in_request(self):
        msgs = [{"role": "user", "content": "x"}]
        _, _, body = runner.build_request(runner.CONFIGS["llama-3.3-70b"], "openrouter", msgs)
        self.assertEqual(body["provider"], {"require_parameters": True, "order": ["deepinfra/turbo", "parasail/fp8"], "allow_fallbacks": False})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        _, _, body = runner.build_request(runner.CONFIGS["gpt-oss-120b-high"], "openrouter", msgs)
        self.assertEqual(body["reasoning"], {"effort": "high"})
        url, _, body = runner.build_request(runner.CONFIGS["gpt-4.1-mini"], "native", msgs)
        self.assertEqual((url, body["model"], body["max_completion_tokens"]), (runner.OPENAI_URL, "gpt-4.1-mini-2025-04-14", 400))
        self.assertNotIn("provider", body)
        _, _, body = runner.build_request(runner.CONFIGS["gpt-4.1-mini"], "openrouter", msgs)
        self.assertEqual(body["provider"]["order"], ["openai", "azure"])
        _, _, body = runner.build_request(runner.CONFIGS["claude-haiku-4.5"], "openrouter", msgs)
        self.assertEqual((body["model"], body["provider"]["order"], body["temperature"]), ("anthropic/claude-haiku-4.5", ["anthropic", "amazon-bedrock"], 1.0))
        self.assertNotIn("reasoning", body)  # extended thinking stays off

    def test_cell_counts(self):
        main = runner.read_jsonl(ROOT / "data/questions/main.jsonl")
        disc = set(json.loads((ROOT / "data/questions/disclaimer_subset.json").read_text()))
        comps = {"belief", "decision", "disclaimer", "followup"}

        def n(cfg, **kw):
            return len(runner.build_cells([cfg], main, comps, disc, T.THRESHOLDS, T.CONDITIONS, **kw))
        # pooled: 300 beliefs + 150 x 5 x 7 decisions + 400 disclaimer cells + second turns (750 x 3 + 400)
        self.assertEqual(n("qwen3-235b"), 300 + 5250 + 400 + 2250 + 400)
        self.assertEqual(n("qwen3-235b"), 8600)
        self.assertEqual(n("gpt-oss-120b-high", e1_qids=disc), 80 + 1400 + 600)  # subset: no disclaimer cells
        self.assertEqual(n("claude-haiku-4.5", e1_qids=disc), 2080)
        self.assertEqual(n("gpt-oss-120b-low", e1_qids=disc), 8600)  # pooled configurations are never restricted
        total = sum(n(c, e1_qids=disc) for c in runner.CONFIGS)
        self.assertEqual(total, 47160)


class MockRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}

    def tearDown(self):
        self.tmp.cleanup()
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def run_once(self, extra=()):
        args = ["--split", "pilot", "--configs", "gpt-4.1-mini,llama-3.3-70b", "--run", "t", "--limit-questions", "2",
                "--budget", "5", "--mock", "--runs-dir", self.tmp.name, "--concurrency", "3", *extra]
        with redirect_stdout(io.StringIO()) as out:
            runner.main(args)
        return out.getvalue()

    def test_fallback_retries_and_resume(self):
        self.run_once()
        rows = {c: [json.loads(l) for l in (Path(self.tmp.name) / "t" / f"{c}.jsonl").read_text().splitlines()]
                for c in ("gpt-4.1-mini", "llama-3.3-70b")}
        # 2 questions x (2 beliefs + 35 decisions) per config
        self.assertEqual({c: len(r) for c, r in rows.items()}, {"gpt-4.1-mini": 74, "llama-3.3-70b": 74})
        self.assertTrue(all(r["status"] == "ok" for rs in rows.values() for r in rs), "503s should be retried")
        g = rows["gpt-4.1-mini"]
        routes = [r["route"] for r in g]
        self.assertGreaterEqual(routes.count("native"), 1)
        self.assertLessEqual(routes.count("native"), 5)  # mock quota: 5 native calls
        self.assertGreater(routes.count("openrouter"), 40)
        self.assertTrue(any(r["fallback"] == "quota" for r in g))
        self.assertTrue(all(r["route"] == "openrouter" for r in rows["llama-3.3-70b"]))
        self.assertTrue(all(r["cost_source"] == "computed_list_price" for r in g if r["route"] == "native"))
        self.assertTrue(all(r["request_params"]["provider"]["allow_fallbacks"] is False for r in rows["llama-3.3-70b"]))
        self.assertTrue(any(len(r["attempts"]) > 1 for rs in rows.values() for r in rs), "some calls should show a retry")
        valid = sum(r["parse"]["valid"] for rs in rows.values() for r in rs)
        self.assertEqual(valid, 148)
        meta = [json.loads(l) for l in (Path(self.tmp.name) / "t" / "run_meta.jsonl").read_text().splitlines()]
        self.assertEqual(len(meta[-1]["route_switches"]), 1)
        self.assertTrue(all(r["template_version"] == T.TEMPLATE_VERSION for rs in rows.values() for r in rs))
        # resume: nothing left to call
        out = self.run_once()
        self.assertIn("148 already have a model response, 0 second turns not applicable (invalid first reply), 0 to call", out)

    def test_second_turn_follows_first_and_resumes(self):
        self.run_once(["--components", "belief,decision,followup"])
        d = Path(self.tmp.name) / "t"
        for c in ("gpt-4.1-mini", "llama-3.3-70b"):
            rows = [json.loads(l) for l in (d / f"{c}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 74 + 30)  # + 2 questions x 5 thresholds x (N, IB, IN)
            first = {r["cell_id"]: r for r in rows if r["kind"] == "decision"}
            fus = [r for r in rows if r["kind"] == "followup"]
            self.assertEqual({r["condition"] for r in fus}, {"N", "IB", "IN"})
            for r in fus:
                p = first[r["parent"]]
                self.assertEqual(r["messages"][:2], p["messages"])
                self.assertEqual(r["messages"][2], {"role": "assistant", "content": p["content"]})
                self.assertEqual(r["messages"][3]["content"], T.followup_user(r["condition"], p["json_order"]))
                self.assertEqual(r["json_order"], p["json_order"])
                self.assertNotIn("parent_content", r)
                self.assertGreaterEqual(r["time_utc"], p["time_utc"])
        # resume after losing the second turns: they are rebuilt from the logged first turns, and nothing else is called
        f = d / "llama-3.3-70b.jsonl"
        f.write_text("".join(l + "\n" for l in f.read_text().splitlines() if '"kind": "followup"' not in l))
        out = self.run_once(["--components", "belief,decision,followup"])
        self.assertIn("30 to call", out)
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        self.assertEqual(sum(r["kind"] == "followup" for r in rows), 30)
        self.assertTrue(all(r["messages"][2]["role"] == "assistant" for r in rows if r["kind"] == "followup"))

    def test_main_run_requires_e1_subset(self):
        base = ["--split", "main", "--configs", "gpt-oss-120b-high", "--run", "m", "--budget", "1", "--mock", "--dry-run",
                "--runs-dir", self.tmp.name]
        with self.assertRaises(SystemExit) as e:
            runner.main(base)
        self.assertIn("--e1-subset", str(e.exception.code))
        with redirect_stdout(io.StringIO()) as out:
            runner.main(base + ["--e1-subset"])
        self.assertIn("1480 cells", out.getvalue())
        with redirect_stdout(io.StringIO()):  # the pilot split (smoke tests) is not restricted
            runner.main(["--split", "pilot", "--configs", "all", "--run", "s", "--budget", "1", "--mock", "--dry-run",
                         "--runs-dir", self.tmp.name, "--limit-questions", "3", "--thresholds", "0.5", "--conditions", "N,IB,LG"])

    def test_resume_refuses_other_template_version(self):
        d = Path(self.tmp.name) / "old"
        d.mkdir()
        (d / "llama-3.3-70b.jsonl").write_text(json.dumps({"cell_id": "x", "status": "ok", "template_version": "old", "cost_usd": 0.0}) + "\n")
        with self.assertRaises(SystemExit) as e:
            runner.main(["--split", "pilot", "--configs", "llama-3.3-70b", "--run", "old", "--budget", "1", "--mock",
                         "--runs-dir", self.tmp.name, "--limit-questions", "1"])
        self.assertIn("Use a new --run name", str(e.exception.code))


if __name__ == "__main__":
    unittest.main()
