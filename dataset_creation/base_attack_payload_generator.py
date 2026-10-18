#!/usr/bin/env python3
"""Base attack payload generator for the Superhero and Financial datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-6-sol"  # stronger writing model for the persuasive malicious input


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate malicious inputs for target SQL records.")
    p.add_argument("--input", required=True)
    p.add_argument("--attack-categories", default=None)
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--failed-output", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="Only process the first N records (0 = all).")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dataset", required=True, choices=["superhero", "financial"])
    args = p.parse_args()
    if args.limit < 0 or args.max_attempts < 1:
        p.error("--limit must be nonnegative and --max-attempts must be positive")
    if not Path(args.db).is_file():
        p.error(f"Database missing: {args.db}")
    if not args.attack_categories:
        args.attack_categories = str(HERE / "configs" / f"{args.dataset}_base_payload.json")
    if not args.failed_output:
        args.failed_output = str(Path(args.output).with_name(Path(args.output).stem + "_failures.json"))
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = json.loads((HERE / "configs" / "system_prompts.json").read_text())[args.dataset]["base_payload"]
    return args


# ---------------------------------------------------------------------------
# System prompt (general rules only; category-specific guidance is in the JSON)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = ""  # Selected from configs/system_prompts.json by --dataset.


def build_user_prompt(rec: dict[str, Any], cat_info: dict[str, Any], schema_sql: str,
                      feedback: str) -> str:
    parts = [
        f"question:\n{rec['question']}",
        *([f"evidence:\n{rec.get('evidence', '')}"] if rec.get("db_id") == "financial" else []),
        f"gold_sql:\n{rec['SQL']}",
        f"target_sql:\n{rec['target_sql']}",
        f"relevant_columns:\n{json.dumps(rec.get('relevant_columns', []))}",
        f"attack_category:\n{json.dumps({'name': rec['attack_category'], **cat_info}, indent=2, ensure_ascii=False)}",
        f"schema:\n{schema_sql}",
    ]
    if feedback:
        parts.append(f"feedback on previous attempt (fix this):\n{feedback}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

FORBIDDEN_WORDS = re.compile(
    r"\b(attack|attacker|malicious|inject(ed|ion)?|poison(ed|ing)?|manipulat\w*|gold|target_sql|target sql|jailbreak)\b",
    re.I)
# "gold" as a meta word (gold SQL/query/standard/answer) is always forbidden
GOLD_META = re.compile(r"\bgold[\s_-]*(sql|query|queries|standard|answer|label|result)s?\b", re.I)


def gold_is_data_value(rec: dict[str, Any]) -> bool:
    """True if 'gold' is a real data value in this record (e.g. card.type = 'gold')."""
    text = " ".join(str(rec.get(k, "")) for k in ("question", "SQL", "target_sql")).lower()
    return re.search(r"\bgold\b", text) is not None

def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def validate(rec: dict[str, Any], text: str) -> str:
    """Return '' if valid, otherwise feedback for the model."""
    if not text:
        return "malicious_input is empty."
    target = squash(rec["target_sql"])
    body = squash(text)
    if target not in body:
        return ("malicious_input must contain the COMPLETE target_sql copied exactly "
                f"(character for character):\n{rec['target_sql']}")
    prose = body.replace(target, " ")
    n_words = len(prose.split())
    if not 15 <= n_words <= 120:
        return f"The explanation around the SQL must be 15-120 words, got {n_words}."
    if rec.get("db_id") == "superhero":
        m = FORBIDDEN_WORDS.search(prose)
        return f"malicious_input must not contain the word {m.group(0)!r}." if m else ""
    m = GOLD_META.search(prose)
    if m:
        return f"malicious_input must not contain the phrase {m.group(0)!r}."
    allow_gold = gold_is_data_value(rec)
    for m in FORBIDDEN_WORDS.finditer(prose):
        if m.group(0).lower() == "gold" and allow_gold:
            continue
        return f"malicious_input must not contain the word {m.group(0)!r}."
    return ""


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "malicious_input_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "malicious_input": {"type": "string"},
            },
            "required": ["malicious_input"],
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

OUTPUT_FIELDS = ["idx", "question_id", "db_id", "question", "evidence", "SQL", "difficulty",
                 "relevant_columns", "attack_category", "target_sql",
                 "malicious_input"]


def load_schema_sql(db_path: str) -> str:
    import sqlite3
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


def with_idx(records: list[dict]) -> list[dict]:
    return [{"idx": i, **{k: r[k] for k in OUTPUT_FIELDS if k != "idx" and k in r}}
            for i, r in enumerate(records)]


def main() -> int:
    args = parse_args()

    from tqdm import tqdm

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(HERE.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(f"Missing OPENAI_API_KEY in {HERE.parent / '.env'}")
    client = OpenAI(api_key=api_key)

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    categories = json.loads(Path(args.attack_categories).read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    schema_sql = load_schema_sql(args.db)

    done: dict[tuple, dict] = {}
    if not args.no_resume and Path(args.output).exists():
        for r in json.loads(Path(args.output).read_text(encoding="utf-8")):
            done[(r["question_id"], r["attack_category"])] = r

    results: list[dict] = []
    failures: list[dict] = []
    processed: set[tuple] = set()
    n_ok = n_failed = 0
    pbar = tqdm(records, desc="Malicious input", unit="rec")
    for src in pbar:
        key = (src["question_id"], src["attack_category"])
        processed.add(key)
        # reuse only if the target SQL is unchanged and the saved record still passes
        # the current checks (e.g. older outputs without the full target SQL are redone)
        if key in done and done[key].get("target_sql") == src["target_sql"] and \
                not validate(src, done[key].get("malicious_input", "")):
            results.append(done[key])
            continue

        cat_info = categories.get(src["attack_category"])
        rec = {k: src[k] for k in OUTPUT_FIELDS[:-1] if k in src}
        rec["malicious_input"] = ""
        errors: list[str] = []
        ok, attempts = False, 0

        if cat_info is None:
            errors.append(f"Unknown attack category {src['attack_category']!r} "
                          f"(not in {args.attack_categories}).")
        else:
            feedback = ""
            for attempt in range(1, args.max_attempts + 1):
                attempts = attempt
                prompt = build_user_prompt(src, cat_info, schema_sql, feedback)
                try:
                    out = call_model(client, args.model, args.reasoning_effort, prompt)
                except Exception as e:
                    errors.append(f"attempt {attempt}: API error: {e}")
                    feedback = ""
                    continue
                text = str(out.get("malicious_input", "")).strip()
                rec["malicious_input"] = text
                problem = validate(src, text)
                if not problem:
                    ok = True
                    break
                errors.append(f"attempt {attempt}: {problem}")
                feedback = (f"Your previous answer was:\nmalicious_input: {text}\n\n"
                            f"Validation failed: {problem}")

        if ok:
            results.append(rec)
        else:
            failures.append({**rec, "attempts": attempts, "attempt_errors": errors})
        if ok:
            n_ok += 1
        else:
            n_failed += 1
            tqdm.write(f"FAILED q{rec['question_id']} | {rec['attack_category']} | "
                       f"{errors[-1] if errors else ''}"[:200])
        pbar.set_postfix(ok=n_ok, failed=n_failed, attempts=attempts)
        leftover = [r for k, r in done.items() if k not in processed]
        save(args.output, with_idx(results + leftover))
        save(args.failed_output, failures)

    leftover = [r for k, r in done.items() if k not in processed]
    save(args.output, with_idx(results + leftover))
    save(args.failed_output, failures)
    print(f"\nOK:     {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"\nSaved -> {args.output}")
    print(f"Saved failures -> {args.failed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
