"""Fetch public safety benchmark datasets and convert them to the
HumaneProxy evaluation format ({"message": ..., "expected": ...}).

Datasets are downloaded at runtime rather than committed to the
repository (they are third-party content with their own licenses):

- XSTest (Rottger et al., 2024, CC-BY-4.0) — 250 superficially alarming
  but SAFE prompts ("How can I kill a Python process?") plus 200 unsafe
  contrast prompts. The safe half measures over-flagging (false
  positives); output: evals/xstest_safe.json
- SimpleSafetyTests (Vidgen et al., 2023) — 100 clearly unsafe prompts
  across five harm areas. Measures harm recall; harm areas are mapped to
  HumaneProxy categories (suicide/self-harm -> self_harm, everything
  else -> criminal_intent). Output: evals/sst.json

Usage:
    python evals/fetch.py            # writes both JSON files next to this script
    hp benchmark --dataset evals/xstest_safe.json
    hp benchmark --dataset evals/sst.json
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import httpx

EVALS_DIR = Path(__file__).resolve().parent

XSTEST_URL = (
    "https://raw.githubusercontent.com/paul-rottger/"
    "exaggerated-safety/main/xstest_prompts.csv"
)
SST_URL = (
    "https://huggingface.co/datasets/Bertievidgen/"
    "SimpleSafetyTests/resolve/main/sst_test_cases.csv"
)

# SimpleSafetyTests harm areas -> HumaneProxy categories.
SST_CATEGORY_MAP = {
    "Suicide, Self-Harm, and Eating Disorders": "self_harm",
    "Physical Harm and Violence": "criminal_intent",
    "Illegal and Highly Regulated items": "criminal_intent",
    "Scams and Fraud": "criminal_intent",
    "Child Abuse": "criminal_intent",
}


def _download_csv(url: str) -> list[dict]:
    print(f"  fetching {url}")
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def fetch_xstest_safe() -> Path:
    """XSTest safe prompts -> false-positive-rate benchmark."""
    rows = _download_csv(XSTEST_URL)
    cases = [
        {
            "message": row["prompt"],
            "expected": "safe",
            "source": f"xstest:{row['type']}",
        }
        for row in rows
        if row.get("label") == "safe" and row.get("prompt")
    ]
    out = EVALS_DIR / "xstest_safe.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"  wrote {out} ({len(cases)} safe prompts)")
    return out


def fetch_sst() -> Path:
    """SimpleSafetyTests -> harm-recall benchmark."""
    rows = _download_csv(SST_URL)
    cases = []
    for row in rows:
        area = (row.get("harm_area") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        if not prompt or area not in SST_CATEGORY_MAP:
            continue
        cases.append(
            {
                "message": prompt,
                "expected": SST_CATEGORY_MAP[area],
                "source": f"sst:{area}",
            }
        )
    out = EVALS_DIR / "sst.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"  wrote {out} ({len(cases)} unsafe prompts)")
    return out


def main() -> int:
    print("Fetching benchmark datasets...")
    try:
        fetch_xstest_safe()
        fetch_sst()
    except httpx.HTTPError as exc:
        print(f"  [ERROR] download failed: {exc}")
        return 1
    print("Done. Run e.g.:")
    print("  hp benchmark --dataset evals/xstest_safe.json")
    print("  hp benchmark --dataset evals/sst.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
