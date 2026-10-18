import json
import re
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    LLMs sometimes wrap JSON with extra text.
    This extracts the first top level JSON object by brace matching.
    """
    if not text:
        return None

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1].strip()
                return safe_json_loads(chunk)
    return None


def is_read_only_sql(sql: str) -> bool:
    s = (sql or "").strip().lower()
    return s.startswith("select") or s.startswith("with")


def split_sql_statements(sql: str) -> List[str]:
    """
    Split by semicolons not inside quotes.
    Used to enforce single statement if you want.
    """
    sql = (sql or "").strip()
    if not sql:
        return []

    out, buf = [], []
    in_single, in_double = False, False

    for ch in sql:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double

        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(ch)

    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out