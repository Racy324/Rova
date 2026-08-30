from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .report import render_report
from .runner import default_root, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed Rova research-agent evaluation.")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--case", action="append", dest="case_ids", choices=tuple(f"C0{index}" for index in range(1, 9)))
    args = parser.parse_args()
    summary = asyncio.run(run_suite(
        repetitions=1 if args.mode == "smoke" else 3,
        root=args.root,
        case_ids=tuple(args.case_ids) if args.case_ids else None,
    ))
    report = render_report(args.root / "results")
    print(f"completed {summary['run_count']} runs; report: {report}")


if __name__ == "__main__":
    main()
