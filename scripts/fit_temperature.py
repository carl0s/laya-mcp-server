"""Fit calibration temperatures for Laya from labeled predictions.

Input: a JSONL file, one line per answered question, for example
  {"type": "choice", "probabilities": {"pass": 0.91, "fail": 0.09}, "expected": "fail"}
  {"type": "noul", "noul": 0.83, "expected": true}

Get the probabilities by calling laya_classify_many with include_all=true on items
you have labeled yourself, then write one line per (item, question).

Output: a calibration JSON for CALIBRATION_PATH, with one temperature per
"choice:<n_options>" group and one for "noul", plus accuracy and expected
calibration error (ECE) before and after.

Usage: python scripts/fit_temperature.py labeled.jsonl > calibration.json
"""

import json
import math
import sys
from collections import defaultdict

GRID = [round(0.5 + 0.05 * i, 2) for i in range(191)]  # 0.5 ... 10.0


def scale_choice(probs, t):
    s = {k: max(p, 1e-12) ** (1 / t) for k, p in probs.items()}
    z = sum(s.values())
    return {k: v / z for k, v in s.items()}


def scale_noul(p, t):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-math.log(p / (1 - p)) / t))


def points(rows, qtype, t):
    """(confidence of predicted answer, was it correct, prob assigned to the truth)."""
    out = []
    for r in rows:
        if qtype == "choice":
            probs = scale_choice(r["probabilities"], t)
            pred = max(probs, key=probs.get)
            out.append((probs[pred], pred == r["expected"], probs.get(r["expected"], 1e-12)))
        else:
            q = scale_noul(r["noul"], t)
            truth = bool(r["expected"])
            out.append((max(q, 1 - q), (q >= 0.5) == truth, q if truth else 1 - q))
    return out


def nll(pts):
    return -sum(math.log(max(p, 1e-12)) for _, _, p in pts) / len(pts)


def ece(pts, bins=10):
    total, err = len(pts), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok, _ in pts if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            err += len(sel) / total * abs(sum(ok for _, ok in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return round(err, 4)


def main(path):
    groups = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                key = f"choice:{len(r['probabilities'])}" if r["type"] == "choice" else "noul"
                groups[key].append(r)

    temps, report = {}, {}
    for key, rows in sorted(groups.items()):
        qtype = "choice" if key.startswith("choice") else "noul"
        best = min(GRID, key=lambda t: nll(points(rows, qtype, t)))
        before, after = points(rows, qtype, 1.0), points(rows, qtype, best)
        temps[key] = best
        report[key] = {
            "n": len(rows),
            "accuracy": round(sum(ok for _, ok, _ in before) / len(rows), 4),
            "ece_before": ece(before),
            "ece_after": ece(after),
        }
        if len(rows) < 50:
            report[key]["warning"] = "fewer than 50 examples, the temperature is not reliable"

    json.dump({"temperatures": temps, "report": report}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
