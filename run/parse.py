"""Parse model replies. The analysis re-parses the raw logged text with these functions.

- JSON is tried first (the whole reply, then any embedded object), then a regex fallback.
- `probability` must be a number from 0 to 100 and is read as a percentage, so 0.5 means 0.5%.
  Values between 0 and 1 are flagged as `below_one`.
- `recommendation`: "buy" -> 1, "don't buy", "do not buy" or "dont buy" -> 0 (case and punctuation ignored).
- A reply is valid only if every required field parses. Nothing is imputed.
- The `reasoning` field is never parsed. When the reply is not clean JSON, its text is blanked out
  before the fallbacks run, so a field is never read from inside the reasoning.
"""
import json
import re

REC_MAP = {"buy": 1, "don't buy": 0, "do not buy": 0, "dont buy": 0}
PROB_RX = re.compile(r"probability[\"']?\s*[:=]\s*[\"']?\s*(-?\d+(?:\.\d+)?)\s*%?", re.I)
REC_RX = re.compile(r"recommendation[\"']?\s*[:=]\s*[\"']?\s*(do not buy|don[’']t buy|dont buy|buy)\b", re.I)
REASONING_VALUE_RX = re.compile(r"""(["']?reasoning["']?\s*:\s*)"(?:[^"\\]|\\.)*(?:"|\\?\Z)""", re.I | re.S)
_DECODER = json.JSONDecoder()


def mask_reasoning(text: str) -> str:
    """Replace the value of every `reasoning` field with an empty string."""
    return REASONING_VALUE_RX.sub(lambda m: m.group(1) + '""', text)


def extract_reasoning(content):
    """The reasoning text of a decision reply, or None."""
    if not isinstance(content, str):
        return None
    try:
        obj = json.loads(content)
    except ValueError:
        return None
    if isinstance(obj, dict):
        v = {str(k).strip().lower(): v for k, v in obj.items()}.get("reasoning")
        return v if isinstance(v, str) and v.strip() else None
    return None


def _norm_rec(v):
    if not isinstance(v, str):
        return None
    s = v.replace("’", "'").replace("‘", "'").lower()
    s = re.sub(r"\s+", " ", s).strip(" \t\n.!,;:\"`")
    return REC_MAP.get(s)


def _number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*%?\s*", v)
        return float(m.group(1)) if m else None
    return None


def _probability(v):
    """Returns (value in [0, 100] or None, below_one flag, error or None)."""
    x = _number(v)
    if x is None:
        return None, False, "probability_not_a_number"
    if not 0 <= x <= 100:
        return None, False, "probability_out_of_range"
    return x, 0 < x < 1, None


def _json_objects(text: str):
    """Yield (method, dict) for the whole reply if it is a JSON object, else for each embedded object."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            yield "json", obj
            return
    except ValueError:
        pass
    i = text.find("{")
    while i != -1:
        try:
            obj, end = _DECODER.raw_decode(text, i)
            if isinstance(obj, dict):
                yield "json_embedded", obj
            i = text.find("{", end)
        except ValueError:
            i = text.find("{", i + 1)


def _parse(content, need_rec: bool) -> dict:
    out = {"valid": False, "probability": None, "recommendation": None, "method": None, "below_one": False, "error": None, "n_json_objects": 0}
    if not isinstance(content, str) or not content.strip():
        out["error"] = "empty_response"
        return out
    keys = ("probability", "recommendation") if need_rec else ("probability",)
    try:
        whole_json = isinstance(json.loads(content), dict)
    except ValueError:
        whole_json = False
    # A reply that is valid JSON as a whole is read as is, the fallbacks read it with the reasoning blanked out.
    masked = mask_reasoning(content)
    objs = [(m, {str(k).strip().lower(): v for k, v in o.items()}) for m, o in _json_objects(content if whole_json else masked)]
    content = masked
    out["n_json_objects"] = len(objs)
    chosen = next(((m, o) for m, o in objs if all(k in o for k in keys)), None)
    chosen = chosen or next(((m, o) for m, o in objs if any(k in o for k in keys)), None)
    raw_p = raw_r = None
    if chosen:
        out["method"], o = chosen
        raw_p, raw_r = o.get("probability"), o.get("recommendation")
    if raw_p is None and (m := PROB_RX.search(content)):
        raw_p, out["method"] = m.group(1), out["method"] or "regex"
    if need_rec and raw_r is None and (m := REC_RX.search(content)):
        raw_r, out["method"] = m.group(1), out["method"] or "regex"
    if raw_p is None:
        out["error"] = "probability_missing"
        return out
    p, out["below_one"], err = _probability(raw_p)
    if err:
        out["error"] = err
        return out
    out["probability"] = p
    if need_rec:
        r = _norm_rec(raw_r)
        if r is None:
            out["error"] = "recommendation_missing" if raw_r is None else "recommendation_unrecognised"
            return out
        out["recommendation"] = r
    out["valid"] = True
    return out


def parse_belief(content) -> dict:
    return _parse(content, need_rec=False)


def parse_decision(content) -> dict:
    return _parse(content, need_rec=True)
