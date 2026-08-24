#!/usr/bin/env python3
"""Concurrency-safe JSONL manifest writer for ``run_dr02_batch.sh``."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path


def record(args: argparse.Namespace) -> int:
    item = {
        "run_id": args.run_id,
        "status": args.status,
        "input": args.input,
        "relative_input": args.relative_input,
        "normalized_output": args.normalized_output,
        "keypoints_output": args.keypoints_output,
        "robot_motion_output": args.robot_motion_output,
        "npz_output": args.npz_output,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "duration": float(args.duration),
        "exit_code": int(args.exit_code),
        "failure_reason": args.failure_reason or None,
        "log_path": args.log_path,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return 0


def summary(args: argparse.Namespace) -> int:
    """Read a run's records while holding the same lock used by ``record``."""
    counts = {"success": 0, "failed": 0, "skipped": 0}
    manifest = Path(args.manifest)
    with manifest.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        handle.seek(0)
        for line in handle:
            item = json.loads(line)
            if item.get("run_id") == args.run_id and item.get("status") in counts:
                counts[item["status"]] += 1
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print(f"success={counts['success']} failed={counts['failed']} skipped={counts['skipped']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read or append DR02 batch manifest records.")
    parser.add_argument("command", choices=("record", "summary"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("success", "failed", "skipped"))
    parser.add_argument("--input", default="")
    parser.add_argument("--relative-input", default="")
    parser.add_argument("--normalized-output", default="")
    parser.add_argument("--keypoints-output", default="")
    parser.add_argument("--robot-motion-output", default="")
    parser.add_argument("--npz-output", default="")
    parser.add_argument("--started-at", default="")
    parser.add_argument("--finished-at", default="")
    parser.add_argument("--duration", default="0")
    parser.add_argument("--exit-code", default="0")
    parser.add_argument("--failure-reason", default="")
    parser.add_argument("--log-path", default="")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(record(parsed) if parsed.command == "record" else summary(parsed))
