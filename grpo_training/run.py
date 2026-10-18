"""Run one or more attack cases, with a fresh training process per case."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "qwen", "secalign", "gpt", "gemini")
REQUIRED = ("idx", "question_id", "db_id", "question", "SQL", "target_sql",
            "attack_category", "relevant_columns", "malicious_input")


def select_records(path, idx=None, limit=0, start_idx=None, end_idx=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [data] if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Input must be a record or a list of records.")
    selected, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every input record must be an object.")
        missing = [key for key in REQUIRED if key not in record]
        if missing:
            raise ValueError(f"Record missing required fields: {missing}")
        number = record["idx"]
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError("Each idx must be an integer.")
        if number in seen:
            raise ValueError(f"Duplicate idx: {number}")
        seen.add(number)
        if idx is not None and number != idx:
            continue
        if start_idx is not None and number < start_idx:
            continue
        if end_idx is not None and number > end_idx:
            continue
        db_id = record["db_id"]
        if not isinstance(db_id, str) or not db_id or Path(db_id).name != db_id:
            raise ValueError("db_id must be a plain database name.")
        columns = record["relevant_columns"]
        if not isinstance(columns, list) or not columns or not all(
            isinstance(col, str) and len(col.split(".")) == 2 for col in columns
        ):
            raise ValueError(f"idx {number}: expected nonempty table.column references")
        selected.append(record)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No cases matched the selection.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=ROOT / "database")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--idx", type=int)
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", type=Path, help="JSON overrides for training settings")
    parser.add_argument("--victim-model", help="Override local, GPT, or Gemini victim model ID")
    parser.add_argument("--validate-only", action="store_true", help="No model loading or API calls")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.model == "secalign" and args.victim_model:
        parser.error("SecAlign requires its configured Llama base and adapter; omit --victim-model")
    input_path = args.input.resolve()
    db_dir = args.db_dir.resolve()
    records = select_records(input_path, args.idx, args.limit, args.start_idx, args.end_idx)
    for record in records:
        path = db_dir / f"{record['db_id']}.sqlite"
        if not path.is_file():
            parser.error(f"Database missing: {path}")
    if args.config:
        values = json.loads(args.config.read_text())
        allowed = json.loads((ROOT / "grpo_training/configs" / f"{args.model}.json").read_text())
        if not isinstance(values, dict) or set(values) - set(allowed):
            parser.error("--config contains unknown training settings")
    if args.validate_only:
        print(json.dumps({"model": args.model, "cases": len(records),
                          "databases": sorted({r['db_id'] for r in records}),
                          "status": "valid; no inference or training performed"}, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or ROOT / "outputs" / args.model / stamp).resolve()
    if args.worker:
        if len(records) != 1:
            parser.error("Worker requires exactly one case")
        os.environ["ARTIFACT_RUN_DIR"] = str(output)
        os.environ["ARTIFACT_DATABASE_DIR"] = str(db_dir)
        os.environ["NL2SQL_SQLGEN_DEFENSE"] = "0"
        if args.config:
            os.environ["ARTIFACT_TRAINING_CONFIG"] = str(args.config.resolve())
        if args.victim_model:
            os.environ["ARTIFACT_VICTIM_MODEL"] = args.victim_model
        # Imports are intentionally deferred until after argument validation.
        os.chdir(ROOT)
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        module = importlib.import_module(f"grpo_training.{args.model}")
        kwargs = {"victim_model": args.victim_model} if args.model in ("gpt", "gemini") else {}
        module.run_experiment(records[0], **kwargs)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "input": str(input_path), "examples": []}
    # Each process owns one victim and attacker, preventing cross-case state reuse.
    for record in records:
        case_dir = output / f"example_{record['idx']:04d}_{record['db_id']}"
        if case_dir.exists() and any(case_dir.iterdir()):
            parser.error(f"Output already exists: {case_dir}. Choose a fresh output directory.")
        case_dir.mkdir(parents=True, exist_ok=True)
        case_path = case_dir / "input.json"
        case_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        command = [sys.executable, "-m", "grpo_training.run", "--worker", "--model", args.model,
                   "--input", str(case_path), "--db-dir", str(db_dir), "--output-dir", str(case_dir)]
        if args.config:
            command += ["--config", str(args.config.resolve())]
        if args.victim_model:
            command += ["--victim-model", args.victim_model]
        print(f"Running {args.model}: idx={record['idx']}", flush=True)
        with (case_dir / "terminal.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        summary["examples"].append({"idx": record["idx"], "exit_code": completed.returncode,
                                     "result": str(case_dir / "result.json")})
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if completed.returncode:
            print(f"Case failed; see {case_dir / 'terminal.log'}", file=sys.stderr)
    return int(any(item["exit_code"] for item in summary["examples"]))


if __name__ == "__main__":
    raise SystemExit(main())
