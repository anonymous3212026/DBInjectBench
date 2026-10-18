from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

from .llm import LocalChatLLM
from .utils import extract_first_json_object


PLANNER_SYSTEM_WITH_SCHEMA = """
You are the PLANNER in a 3-step agentic NL2SQL system (Planner -> Retriever -> SQL Generator).

You output a retrieval plan only. You do not write SQL. You do not call tools.

You are given the database schema (tables, columns, types). Use EXACT table and column names from the schema.
Do not invent names.

Retriever tools available:
- list_tables()
- describe_table(table)
- foreign_keys(table)
- sample_distinct(table, column, k)
- search_like(table, column, pattern, k)

Goal:
Produce a retrieval plan that helps later agents write correct SQL without guessing. The plan should:
- indicate what schema information is needed (tables, columns, relationships)
- indicate what literal values are likely needed for database grounding using value lookup (distinct values)
- provide a sequence of retrieval actions that will ground the needed literals and confirm join paths.

Output:
Return ONLY valid JSON with this schema (no extra keys):

{
  "main_entity": "what the question is about",
  "retriever_steps": [
    {
      "goal": "short purpose of the step",
      "tool": "list_tables|describe_table|foreign_keys|sample_distinct|search_like",
      "args_template": { "table": "exact_table", "column": "exact_column", "pattern": "%...%", "k": 500 },
      "expected_output": "what the retriever should learn from this step"
    }
  ],
  "value_lookup": [
    {
      "value_mention": "exact phrase from the question",
      "why_lookup_needed": "short reason",
      "lookup_plan": {
        "candidate_tables": ["exact_table"],
        "candidate_columns": ["exact_column"],
        "preferred_method": "sample_distinct|search_like",
        "search_pattern_if_needed": "%...%"
      }
    }
  ]
}

Guidelines:
- Prefer exact names from the schema.
- Use list_tables only if the schema context is empty or clearly incomplete.
- Always include at least one foreign_keys step when multi-table joins are likely.
- Use search_like for names and free text mentions; use sample_distinct for categorical fields.
- Keep steps minimal and targeted.
""".strip()


def _as_str(x: Any, default: str = "") -> str:
    return x if isinstance(x, str) else default


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _default_plan(question: str) -> Dict[str, Any]:
    return {
        "main_entity": "unknown",
        "value_lookup": [],
        "retriever_steps": [
            {
                "goal": "Get all table names to avoid guessing.",
                "tool": "list_tables",
                "args_template": {},
                "expected_output": "List of table names.",
            },
            {
                "goal": "Collect foreign keys for join paths.",
                "tool": "foreign_keys",
                "args_template": {"table": ""},
                "expected_output": "Foreign key relationships for relevant tables.",
            },
        ],
    }


def _normalize_retriever_step(step: Any) -> Dict[str, Any]:
    if not isinstance(step, dict):
        return {
            "goal": "",
            "tool": "list_tables",
            "args_template": {},
            "expected_output": "",
        }

    tool = _as_str(step.get("tool", "list_tables")).strip()
    allowed = {"list_tables", "describe_table", "foreign_keys", "sample_distinct", "search_like"}
    if tool not in allowed:
        tool = "list_tables"

    goal = _as_str(step.get("goal", "")).strip()
    expected_output = _as_str(step.get("expected_output", "")).strip()

    args = step.get("args_template", {})
    if not isinstance(args, dict):
        args = {}

    cleaned: Dict[str, Any] = {}
    for k in ("table", "column", "pattern", "k"):
        if k in args:
            cleaned[k] = args[k]

    # Defensive: strip any accidental placeholders
    for k in list(cleaned.keys()):
        v = cleaned[k]
        if isinstance(v, str) and "?" in v:
            cleaned[k] = v.replace("?", "").strip()

    return {
        "goal": goal,
        "tool": tool,
        "args_template": cleaned,
        "expected_output": expected_output,
    }


def _normalize_value_lookup_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "value_mention": "",
            "why_lookup_needed": "",
            "lookup_plan": {
                "candidate_tables": [],
                "candidate_columns": [],
                "preferred_method": "search_like",
                "search_pattern_if_needed": "",
            },
        }

    value_mention = _as_str(item.get("value_mention", "")).strip()
    why_lookup_needed = _as_str(item.get("why_lookup_needed", "")).strip()

    lp = item.get("lookup_plan", {})
    if not isinstance(lp, dict):
        lp = {}

    candidate_tables = [t.strip() for t in _as_list(lp.get("candidate_tables")) if isinstance(t, str) and t.strip()]
    candidate_columns = [c.strip() for c in _as_list(lp.get("candidate_columns")) if isinstance(c, str) and c.strip()]

    preferred_method = _as_str(lp.get("preferred_method", "search_like")).strip().lower()
    if preferred_method not in {"sample_distinct", "search_like"}:
        preferred_method = "search_like"

    search_pattern = _as_str(lp.get("search_pattern_if_needed", "")).strip()
    if preferred_method == "search_like" and not search_pattern and value_mention:
        search_pattern = f"%{value_mention}%"

    # Defensive: strip placeholders
    candidate_tables = [t.replace("?", "").strip() for t in candidate_tables if t.replace("?", "").strip()]
    candidate_columns = [c.replace("?", "").strip() for c in candidate_columns if c.replace("?", "").strip()]
    search_pattern = search_pattern.replace("?", "").strip()

    return {
        "value_mention": value_mention,
        "why_lookup_needed": why_lookup_needed,
        "lookup_plan": {
            "candidate_tables": candidate_tables,
            "candidate_columns": candidate_columns,
            "preferred_method": preferred_method,
            "search_pattern_if_needed": search_pattern,
        },
    }


def _ensure_min_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools_in = [s.get("tool") for s in steps if isinstance(s, dict)]
    normalized = list(steps)

    # If joins are likely and foreign_keys is missing, add it.
    if "foreign_keys" not in tools_in:
        normalized.append(
            {
                "goal": "Collect foreign keys to determine valid join paths.",
                "tool": "foreign_keys",
                "args_template": {"table": ""},
                "expected_output": "Foreign keys for relevant tables.",
            }
        )
    return normalized


class PlannerAgent:
    def __init__(self, llm: LocalChatLLM, sqlite_db_path: str) -> None:
        self.llm = llm
        self.db_path = sqlite_db_path

    def _load_schema_context(self) -> Dict[str, List[Dict[str, Any]]]:
        schema: Dict[str, List[Dict[str, Any]]] = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

            for row in tables:
                table = row["name"]
                table_ident = table.replace('"', '""')
                cols = conn.execute(f'PRAGMA table_info("{table_ident}")').fetchall()
                schema[table] = [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "notnull": col["notnull"],
                        "pk": col["pk"],
                    }
                    for col in cols
                ]
        return schema

    def run(self, question: str) -> Dict[str, Any]:
        schema_context = self._load_schema_context()

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_WITH_SCHEMA},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "schema_context": schema_context},
                    ensure_ascii=False,
                ),
            },
        ]

        raw = self.llm.chat(messages)
        obj = extract_first_json_object(raw)
        if obj is None or not isinstance(obj, dict):
            return _default_plan(question)

        main_entity = _as_str(obj.get("main_entity", "unknown")).strip() or "unknown"

        steps_raw = _as_list(obj.get("retriever_steps"))
        steps = [_normalize_retriever_step(s) for s in steps_raw]
        if not steps:
            steps = _default_plan(question)["retriever_steps"]

        value_lookup_raw = _as_list(obj.get("value_lookup"))
        value_lookup = [_normalize_value_lookup_item(v) for v in value_lookup_raw]
        value_lookup = [v for v in value_lookup if v.get("value_mention")]

        steps = _ensure_min_steps(steps)

        return {
            "main_entity": main_entity,
            "retriever_steps": steps,
            "value_lookup": value_lookup,
        }
