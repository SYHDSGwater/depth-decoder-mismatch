from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
TEMPLATE = EXP_DIR / "EXP-000-template.md"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "experiment"


def next_id() -> int:
    ids = []
    for p in EXP_DIR.glob("EXP-*.md"):
        m = re.match(r"EXP-(\d+)", p.name)
        if m and int(m.group(1)) != 0:
            ids.append(int(m.group(1)))
    return max(ids, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    n = next_id(); exp_id = f"EXP-{n:03d}"
    path = EXP_DIR / f"{exp_id}-{slugify(args.title)}.md"
    content = TEMPLATE.read_text(encoding="utf-8")
    content = content.replace("EXP-000 — Experiment title", f"{exp_id} — {args.title}", 1)
    path.write_text(content, encoding="utf-8")
    print(path.relative_to(ROOT))

if __name__ == "__main__": main()
