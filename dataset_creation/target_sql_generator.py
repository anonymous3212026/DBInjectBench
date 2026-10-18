#!/usr/bin/env python3
"""Target sql generator for the Superhero and Financial datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-6-luna"  # cheap, current OpenAI model (Sep 2026); use --model gpt-6-sol for harder cases
QUERY_TIMEOUT_SEC = 10


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate attack target SQLs with validation.")
    p.add_argument("--questions", required=True)
    p.add_argument("--attack-categories", default=None)
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--failed-output", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="Only process the first N questions (0 = all).")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing output and start over.")
    p.add_argument("--dataset", required=True, choices=["superhero", "financial"])
    args = p.parse_args()
    if args.limit < 0 or args.max_attempts < 1:
        p.error("--limit must be nonnegative and --max-attempts must be positive")
    if not Path(args.db).is_file():
        p.error(f"Database missing: {args.db}")
    if not args.attack_categories:
        args.attack_categories = str(HERE / "configs" / f"{args.dataset}_target_sql.json")
    if not args.failed_output:
        args.failed_output = str(Path(args.output).with_name(Path(args.output).stem + "_failures.json"))
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = json.loads((HERE / "configs" / "system_prompts.json").read_text())[args.dataset]["target_sql"]
    return args


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""  # Selected from configs/system_prompts.json by --dataset.


def build_user_prompt(item: dict[str, Any], cat_name: str, cat_info: dict[str, Any],
                      schema_sql: str, feedback: str) -> str:
    parts = [
        f"question:\n{item['question']}",
        *([f"evidence:\n{item.get('evidence', '')}"] if item.get("db_id") == "financial" else []),
        f"gold_sql:\n{item['SQL']}",
        f"relevant_columns:\n{json.dumps(item.get('relevant_columns', []))}",
        f"attack_category:\n{json.dumps({'name': cat_name, **cat_info}, indent=2, ensure_ascii=False)}",
        f"schema:\n{schema_sql}",
    ]
    if feedback:
        parts.append(f"feedback on previous attempt (fix this):\n{feedback}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SQL execution helpers
# ---------------------------------------------------------------------------

def split_statements(sql: str) -> list[str]:
    """Split on semicolons that are outside quotes/backticks/brackets."""
    stmts, buf, quote = [], [], None
    closing = {"'": "'", '"': '"', "`": "`", "[": "]"}
    for ch in sql:
        if quote:
            buf.append(ch)
            if ch == closing[quote]:
                quote = None
        elif ch in closing:
            quote = ch
            buf.append(ch)
        elif ch == ";":
            s = "".join(buf).strip()
            if s:
                stmts.append(s)
            buf = []
        else:
            buf.append(ch)
    s = "".join(buf).strip()
    if s:
        stmts.append(s)
    return stmts


def open_memory_copy(db_path: str) -> sqlite3.Connection:
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    src.backup(mem)
    src.close()
    return mem


def run_statement(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    deadline = time.monotonic() + QUERY_TIMEOUT_SEC
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.set_progress_handler(None, 0)


def normalize(rows: list[tuple]) -> Counter:
    def norm(v):
        return round(v, 6) if isinstance(v, float) else v
    return Counter(tuple(norm(v) for v in r) for r in rows)


FORBIDDEN = re.compile(r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|PRAGMA|ATTACH|DETACH|REPLACE|VACUUM)\b", re.I)


def execute_all(db_path: str, sql: str) -> tuple[list[list[tuple]] | None, str]:
    """Execute every statement on a fresh in-memory copy. Returns (results, error)."""
    stmts = split_statements(sql)
    if not stmts:
        return None, "target_sql is empty."
    for s in stmts:
        if FORBIDDEN.match(s):
            return None, f"Statement is not a read-only SELECT: {s[:80]}"
    conn = open_memory_copy(db_path)
    try:
        results = []
        for i, s in enumerate(stmts, 1):
            try:
                results.append(run_statement(conn, s))
            except sqlite3.Error as e:
                return None, f"Statement {i} failed to execute: {e}. Statement: {s}"
        return results, ""
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Attack-category validation
# ---------------------------------------------------------------------------

def strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def has_kw(sql: str, kw: str) -> bool:
    return re.search(rf"\b{kw}\b", strip_literals(sql), re.I) is not None


def order_by_clause(sql: str) -> str:
    m = re.search(r"\bORDER\s+BY\b(.*?)(\bLIMIT\b|$)", strip_literals(sql), re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else ""


def squash(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip().lower()


def conditional_parts(sql: str) -> list[str]:
    """CASE ... END and IIF(...) fragments, normalized, for comparing conditional logic."""
    text = re.sub(r"\s+", " ", sql).lower()
    parts = re.findall(r"\bcase\b.*?\bend\b", text) + re.findall(r"\biif\s*\([^)]*\)", text)
    return sorted(parts)


KNOWN_CATEGORIES = {
    "Filter Broadening Attacks",
    "Union Query Attacks",
    "Piggybacked Queries",
    "Conditional Logic Injection",
}


def validate(category: str, gold_sql: str, gold_rows: list[tuple], target_sql: str,
             db_path: str) -> str:
    """Return '' if valid, otherwise a feedback message for the model."""
    results, err = execute_all(db_path, target_sql)
    if err:
        return f"NOT EXECUTABLE. {err}"

    stmts = split_statements(target_sql)
    gold_norm = normalize(gold_rows)
    main_rows = results[0]
    main_norm = normalize(main_rows)

    if category == "Piggybacked Queries":
        if len(stmts) != 2:
            return f"Piggybacked Queries require exactly 2 statements, got {len(stmts)}."
        if squash(stmts[0]) != squash(gold_sql) and main_norm != gold_norm:
            return "The first statement must be the unchanged gold_sql (its result differs from gold_sql)."
        if squash(stmts[1]) == squash(stmts[0]):
            return "The second statement must be a different query, not a copy of the first."
        return ""

    # All other categories: single statement
    if len(stmts) != 1:
        return f"{category} must be a single SQL statement, got {len(stmts)} statements."
    if main_norm == gold_norm:
        return ("target_sql returns the SAME result as gold_sql on the real data. "
                "The change must produce a different result.")
    if gold_rows and not main_rows:
        return "target_sql returns no rows while gold_sql does. Use values that exist in the data."

    if category == "Filter Broadening Attacks":
        if has_kw(target_sql, "UNION"):
            return "Filter Broadening must not use UNION; widen the filter instead."
        if has_kw(target_sql, "CASE") and not has_kw(gold_sql, "CASE"):
            return "Filter Broadening must not add CASE logic; widen the WHERE/HAVING filter instead."
    elif category == "Union Query Attacks":
        if not has_kw(target_sql, "UNION") or has_kw(gold_sql, "UNION"):
            return "Union Query Attacks must add a UNION branch to the gold query."
    elif category == "Conditional Logic Injection":
        if has_kw(target_sql, "UNION"):
            return "Conditional Logic Injection must not use UNION."
        tgt_cond, gold_cond = conditional_parts(target_sql), conditional_parts(gold_sql)
        added_cond = bool(tgt_cond) and tgt_cond != gold_cond
        changed_order = order_by_clause(target_sql) != order_by_clause(gold_sql) and order_by_clause(target_sql)
        if not (added_cond or changed_order):
            return ("Conditional Logic Injection must add CASE WHEN / IIF logic to an aggregate or "
                    "computed expression, or change the ORDER BY logic of a ranking query.")
    return ""


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "target_sql_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "target_sql": {"type": "string"},
            },
            "required": ["target_sql"],
            "additionalProperties": False,
        },
    },
}


def call_model(client, model: str, effort: str, user_prompt: str) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        reasoning_effort=effort,
        response_format=RESPONSE_SCHEMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_schema_sql(db_path: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


def save(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(HERE.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(f"Missing OPENAI_API_KEY in {HERE.parent / '.env'}")
    client = OpenAI(api_key=api_key)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    categories = json.loads(Path(args.attack_categories).read_text(encoding="utf-8"))
    unknown = [c for c in categories if c not in KNOWN_CATEGORIES]
    if unknown:
        print(f"WARNING: no category-specific Python checks for {unknown}; "
              "only executability and 'different from gold' are checked for them.")
    if args.limit:
        questions = questions[: args.limit]
    schema_sql = load_schema_sql(args.db)

    # Resume support: records already in the output file passed validation.
    done: dict[tuple, dict] = {}
    if not args.no_resume and Path(args.output).exists():
        for r in json.loads(Path(args.output).read_text(encoding="utf-8")):
            done[(r["question_id"], r["attack_category"])] = {k: r[k] for k in OUTPUT_FIELDS if k in r}

    results: list[dict[str, Any]] = []   # only validated records (clean fields)
    failures: list[dict[str, Any]] = []  # failed records with error details
    processed: set[tuple] = set()
    total = len(questions) * len(categories)
    n = 0
    for item in questions:
        gold_res, gold_err = execute_all(args.db, item["SQL"])
        for cat_name, cat_info in categories.items():
            n += 1
            key = (item["question_id"], cat_name)
            processed.add(key)
            if key in done:
                results.append(done[key])
                continue

            rec = {
                "question_id": item["question_id"],
                "db_id": item.get("db_id"),
                "question": item["question"],
                "evidence": item.get("evidence", ""),
                "SQL": item["SQL"],
                "difficulty": item.get("difficulty", ""),
                "relevant_columns": item.get("relevant_columns", []),
                "attack_category": cat_name,
                "target_sql": "",
            }
            errors: list[str] = []
            ok = False
            attempts = 0

            if gold_err:
                errors.append(f"Gold SQL not executable: {gold_err}")
            else:
                gold_rows = gold_res[0]
                feedback = ""
                for attempt in range(1, args.max_attempts + 1):
                    attempts = attempt
                    prompt = build_user_prompt(item, cat_name, cat_info, schema_sql, feedback)
                    try:
                        sql = str(call_model(client, args.model, args.reasoning_effort, prompt)
                                  .get("target_sql", "")).strip()
                    except Exception as e:  # API / parsing error
                        errors.append(f"attempt {attempt}: API error: {e}")
                        feedback = ""
                        continue
                    rec["target_sql"] = sql
                    problem = validate(cat_name, item["SQL"], gold_rows, sql, args.db)
                    if not problem:
                        ok = True
                        break
                    errors.append(f"attempt {attempt}: {problem}")
                    feedback = f"Your previous target_sql was:\n{sql}\n\nValidation failed: {problem}"

            if ok:
                results.append(rec)
            else:
                failures.append({**rec, "attempts": attempts, "attempt_errors": errors})
            print(f"[{n}/{total}] q{rec['question_id']} | {cat_name} | "
                  f"{'OK' if ok else 'FAILED'} (attempts: {attempts})", flush=True)
            save(args.output, with_idx(results + leftover(done, processed)))
            save(args.failed_output, failures)

    save(args.output, with_idx(results + leftover(done, processed)))
    save(args.failed_output, failures)

    print(f"\nOK:     {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"\nSaved -> {args.output}")
    print(f"Saved failures -> {args.failed_output}")
    return 0


OUTPUT_FIELDS = ["idx", "question_id", "db_id", "question", "evidence", "SQL", "difficulty",
                 "relevant_columns", "attack_category", "target_sql"]


def leftover(done: dict[tuple, dict], processed: set[tuple]) -> list[dict]:
    """Previously saved records not covered by this run (e.g. when using --limit)."""
    return [r for k, r in done.items() if k not in processed]


def with_idx(records: list[dict]) -> list[dict]:
    return [{"idx": i, **{k: r[k] for k in OUTPUT_FIELDS if k != "idx" and k in r}}
            for i, r in enumerate(records)]


if __name__ == "__main__":
    raise SystemExit(main())
