from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List


def canonicalize_query_rows(rows: Iterable[tuple]) -> List[str]:
    """
    Convert query rows into stable comparable strings.

    Row order and duplicate rows are intentionally preserved.
    """
    canonical_rows: List[str] = []
    for row in rows:
        normalized_row: List[Any] = []
        for value in row:
            if isinstance(value, bytes):
                normalized_row.append({"__bytes_hex__": value.hex()})
            else:
                normalized_row.append(value)
        canonical_rows.append(json.dumps(normalized_row, sort_keys=True, default=str))
    return canonical_rows


def ordered_overlap_count(pred_rows: Iterable[tuple], target_rows: Iterable[tuple]) -> int:
    pred_canon = canonicalize_query_rows(pred_rows)
    target_canon = canonicalize_query_rows(target_rows)
    return sum(
        1
        for pred_row, target_row in zip(pred_canon, target_canon)
        if pred_row == target_row
    )


def multiset_overlap_count(pred_rows: Iterable[tuple], target_rows: Iterable[tuple]) -> int:
    """
    Backward-compatible alias for older tests/imports.

    The current reward is order-sensitive, so this returns ordered overlap.
    """
    return ordered_overlap_count(pred_rows, target_rows)


def compute_f1_overlap(pred_rows: Iterable[tuple], target_rows: Iterable[tuple]) -> float:
    pred_canon = canonicalize_query_rows(pred_rows)
    target_canon = canonicalize_query_rows(target_rows)

    pred_count = len(pred_canon)
    target_count = len(target_canon)

    if target_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    if pred_count == 0:
        return 0.0

    overlap = sum(
        1
        for pred_row, target_row in zip(pred_canon, target_canon)
        if pred_row == target_row
    )

    if overlap == 0:
        return 0.0

    precision = overlap / pred_count
    recall = overlap / target_count
    return 2.0 * precision * recall / (precision + recall)


def execute_sql_readonly(db_path: str | Path, sql_query: str) -> tuple[bool, List[tuple], str]:
    """
    Execute SQL in read-only mode and return rows.

    Returns:
    - ok: whether execution succeeded
    - rows: fetched rows, empty on failure or non-result statements
    - message: diagnostic text
    """
    path = Path(db_path)
    if not path.exists():
        return False, [], f"DB file not found: {path}"

    uri = f"file:{path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return True, rows, "ok"
    except Exception as exc:
        return False, [], f"{type(exc).__name__}: {exc}"


def compute_execution_target_component(
    db_path: str | Path, predicted_sql: str, target_sql: str
) -> tuple[float, bool, bool, bool, str]:
    """
    Compute a non-binary execution-target component.

    Component definition:
    - Order-sensitive F1 row-overlap between predicted SQL output and target SQL output.
    - 0.0 if either query fails to execute.
    - success remains exact-result success for ASR-style reporting.

    Returns:
    - component: float
    - success: exact canonical result match
    - pred_ok: whether predicted SQL executed
    - target_ok: whether target SQL executed
    - debug_message: diagnostic text
    """
    pred_ok, pred_rows, pred_msg = execute_sql_readonly(db_path, predicted_sql)
    target_ok, target_rows, target_msg = execute_sql_readonly(db_path, target_sql)

    if not pred_ok or not target_ok:
        debug = (
            f"Execution failure | pred_ok={pred_ok} ({pred_msg}) | "
            f"target_ok={target_ok} ({target_msg})"
        )
        return 0.0, False, pred_ok, target_ok, debug

    pred_canon = canonicalize_query_rows(pred_rows)
    target_canon = canonicalize_query_rows(target_rows)
    exact_match = pred_canon == target_canon

    component = compute_f1_overlap(pred_rows, target_rows)
    overlap = ordered_overlap_count(pred_rows, target_rows)
    precision = overlap / len(pred_rows) if pred_rows else (1.0 if not target_rows else 0.0)
    recall = overlap / len(target_rows) if target_rows else (1.0 if not pred_rows else 0.0)

    debug = (
        "Execution compare | "
        f"pred_rows={len(pred_rows)} | target_rows={len(target_rows)} | "
        f"overlap={overlap} | precision={precision:.4f} | recall={recall:.4f} | "
        f"f1={component:.4f} | exact_match={exact_match}"
    )
    return component, exact_match, pred_ok, target_ok, debug
