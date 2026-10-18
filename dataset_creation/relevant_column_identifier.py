"""Identify relevant columns, resolve SQL aliases, and retain text-column questions."""
from __future__ import annotations
import argparse
import ast
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
SYSTEM_PROMPT = '\nYou extract relevant database columns for NL2SQL.\n\nInput:\n- question: natural language question\n- sql: SQL query for the same question\n\nTask:\n- Return the list of database columns that are most relevant to answering the question.\n- Prefer fully-qualified names from SQL aliases when possible (for example: superhero.eye_colour_id).\n- Include columns used in SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, and key filter predicates.\n- Keep the list concise and deduplicated.\n\nOutput format:\nReturn ONLY valid JSON in this exact shape:\n{\n  "relevant_columns": ["table_or_alias.column", "..."]\n}\n'
TEXT_TYPE_MARKERS = ("CHAR", "CLOB", "TEXT")
def load_questions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if not isinstance(data, list):
        raise ValueError('Input questions file must contain a list of objects.')
    return data

def extract_sql(item: dict[str, Any]) -> str:
    sql = item.get('sql')
    if isinstance(sql, str) and sql.strip():
        return sql
    sql_upper = item.get('SQL')
    if isinstance(sql_upper, str) and sql_upper.strip():
        return sql_upper
    return ''

def get_relevant_columns(client: Any, model: str, question: str, sql: str) -> list[str]:
    user_prompt = f'question:\n{question}\n\nsql:\n{sql}\n\nReturn the result now.'
    response = client.chat.completions.create(model=model, temperature=0, response_format={'type': 'json_object'}, messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}])
    content = (response.choices[0].message.content or '').strip()
    parsed = json.loads(content)
    columns = parsed.get('relevant_columns', [])
    if not isinstance(columns, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if not isinstance(col, str):
            continue
        c = col.strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def is_text_type(declared_type: str) -> bool:
    return any((m in (declared_type or '').upper() for m in TEXT_TYPE_MARKERS))

def load_schema(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, Any]]]:
    """table(lower) -> column(lower) -> info"""
    schema: dict[str, dict[str, dict[str, Any]]] = {}
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        fk_cols = {r[3].lower() for r in conn.execute(f'PRAGMA foreign_key_list("{t}")')}
        cols = {}
        for (_, name, ctype, _, _, pk) in conn.execute(f'PRAGMA table_info("{t}")'):
            cols[name.lower()] = {'table': t, 'column': name, 'type': ctype, 'is_text': is_text_type(ctype), 'is_key': bool(pk) or name.lower() in fk_cols}
        schema[t.lower()] = cols
    return schema

ALIAS_RE = re.compile('\\b(?:FROM|JOIN)\\s+[`\\"\\[]?(\\w+)[`\\"\\]]?\\s+(?:AS\\s+)?(\\w+)', re.I)

NOT_ALIAS = {'on', 'where', 'inner', 'left', 'right', 'join', 'group', 'order', 'limit', 'union', 'cross', 'natural', 'outer', 'full', 'having', 'using'}

def alias_map(sql: str, schema) -> dict[str, set[str]]:
    """alias(lower) -> set of real table names (lower) it is bound to in the SQL."""
    out: dict[str, set[str]] = {}
    for (table, alias) in ALIAS_RE.findall(sql or ''):
        if table.lower() in schema and alias.lower() not in NOT_ALIAS:
            out.setdefault(alias.lower(), set()).add(table.lower())
    return out

def classify(ref: str, schema, aliases=None) -> tuple[str, dict[str, Any] | None]:
    """Returns (status, info). status in: text, non_text, unknown_table,
    unknown_column, ambiguous_alias, bad_format."""
    if not isinstance(ref, str) or '.' not in ref:
        return ('bad_format', None)
    (table, column) = ref.split('.', 1)
    (table, column) = (table.strip().strip('`"'), column.strip().strip('`"'))
    cols = schema.get(table.lower())
    if cols is None:
        candidates = [t for t in (aliases or {}).get(table.lower(), set()) if column.lower() in schema[t]]
        if len(candidates) > 1:
            return ('ambiguous_alias', None)
        if not candidates:
            return ('unknown_column' if (aliases or {}).get(table.lower()) else 'unknown_table', None)
        cols = schema[candidates[0]]
    info = cols.get(column.lower())
    if info is None:
        return ('unknown_column', None)
    return ('text' if info['is_text'] else 'non_text', info)

def filter_text_columns(records, db):
    """Resolve aliases and discard questions with no injectable text columns."""
    with sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True) as conn:
        schema = load_schema(conn)
    kept, removed, issues = [], [], []
    for record in records:
        aliases = alias_map(extract_sql(record), schema)
        columns = []
        for ref in record.get('relevant_columns', []):
            status, info = classify(ref, schema, aliases)
            if status == 'text':
                columns.append(f"{info['table']}.{info['column']}")
            else:
                issues.append({'question_id': record.get('question_id'), 'column': ref, 'status': status})
        row = dict(record)
        row['relevant_columns'] = list(dict.fromkeys(columns))
        (kept if columns else removed).append(row)
    return kept, {'removed_questions': removed, 'dropped_columns': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--reuse-columns', action='store_true', help='Filter existing relevant_columns without API calls')
    parser.add_argument('--raw-output', type=Path, help='Optional copy before text-column filtering')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f'Database missing: {args.db}')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    records = load_questions(args.input)
    if args.limit:
        records = records[:args.limit]
    client = None
    if not args.reuse_columns:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(Path(__file__).resolve().parents[1] / '.env')
        client = OpenAI()
    enriched = []
    for i, record in enumerate(records, 1):
        row = dict(record)
        sql = extract_sql(row)
        row['SQL'] = sql
        if args.reuse_columns:
            if 'relevant_columns' not in row:
                parser.error('--reuse-columns requires relevant_columns in every record')
        else:
            row['relevant_columns'] = get_relevant_columns(client, args.model, row['question'], sql) if sql else []
        enriched.append(row)
        print(f'[{i}/{len(records)}] processed')
    output, report = filter_text_columns(enriched, args.db)
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    save(args.output, output)
    save(args.report or args.output.with_name(args.output.stem + '_column_report.json'), report)
    if args.raw_output:
        save(args.raw_output, enriched)
    print(f'Kept {len(output)} of {len(records)} questions with relevant text columns.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
