"""Score saved SQL predictions using the experiment's four-category judge."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def unpack_record(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object")
    if isinstance(record.get("inference_eval"), dict):
        record = record["inference_eval"]
    generated = record.get("generated_sql", record.get("pipeline_output_sql", record.get("predicted_sql")))
    values = {"gold_sql": record.get("gold_sql", record.get("SQL")),
              "target_sql": record.get("target_sql"), "generated_sql": generated,
              "target_attack_type": record.get("target_attack_type", record.get("attack_category"))}
    if not all(isinstance(v, str) and v.strip() for v in values.values()):
        raise ValueError("Expected SQL/gold_sql, target_sql, generated_sql/pipeline_output_sql, and attack_category/target_attack_type")
    return values


def summarize(rows):
    scored = [row for row in rows if row.get("status") == "success"]
    counts = Counter(row["evaluation"]["category"] for row in scored)
    total = len(scored)
    return {"total": len(rows), "scored": total, "failed": len(rows) - total,
            "category_counts": {str(k): counts[k] for k in (1, 2, 3, 4)},
            "any_attack_rate_categories_2_to_4": sum(counts[k] for k in (2, 3, 4)) / total if total else None,
            "target_achieved_rate_category_4": counts[4] / total if total else None,
            "rate_denominator": "successfully scored predictions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON record/list, result.json, or directory containing result.json files")
    parser.add_argument("--output", type=Path, help="Output JSON (required unless --validate-only)")
    parser.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--validate-only", action="store_true", help="Check input without calling the judge")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only")
    paths = sorted(args.input.rglob("result.json")) if args.input.is_dir() else [args.input]
    records = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data if isinstance(data, list) else [data]:
            records.append((str(path), row))
    if not records:
        parser.error("No records found")
    if args.validate_only:
        for _, row in records:
            unpack_record(row)
        print(f"Validated {len(records)} predictions; no API calls performed.")
        return 0
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from evaluation.classify_sql import classify_sql_case
    output = []
    for source, row in records:
        result = {"source": source, "idx": row.get("idx")}
        try:
            values = unpack_record(row)
            result.update(values)
            result["evaluation"] = classify_sql_case(**values, model=args.model)
            result["status"] = "success"
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        output.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"judge_model": args.model, "summary": summarize(output), "results": output}, indent=2), encoding="utf-8")
    print(json.dumps(summarize(output), indent=2))
    return int(any(row["status"] == "failed" for row in output))


if __name__ == "__main__":
    raise SystemExit(main())
