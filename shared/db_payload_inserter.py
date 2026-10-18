#!/usr/bin/env python3
"""
SQLite payload insertion helpers for injection benchmarking.

This module provides schema-aware insertion utilities:
- Validate target column is text-like.
- Generate random values for other columns in the row.
- Insert one row per target column.
- Handle single-column foreign keys by sampling valid referenced values.

The main function for your use-case is:
`insert_payload_into_text_columns(...)`
"""

from __future__ import annotations

import random
import sqlite3
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ColumnInfo:
    """
    Basic schema metadata for one SQLite column.
    """

    cid: int
    name: str
    col_type: str
    not_null: bool
    default_value: Optional[str]
    pk_position: int


def _read_table_schema(conn: sqlite3.Connection, table_name: str) -> List[ColumnInfo]:
    """
    Read table schema via PRAGMA table_info.
    """
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    if not rows:
        raise ValueError(f"Table '{table_name}' not found or has no columns.")

    cols: List[ColumnInfo] = []
    for row in rows:
        cols.append(
            ColumnInfo(
                cid=int(row[0]),
                name=str(row[1]),
                col_type=str(row[2] or ""),
                not_null=bool(row[3]),
                default_value=row[4],
                pk_position=int(row[5]),
            )
        )
    return cols


def _is_text_like(col_type: str) -> bool:
    """
    True if declared type is text-like in SQLite terms.
    """
    t = col_type.upper()
    return ("CHAR" in t) or ("CLOB" in t) or ("TEXT" in t)


def _is_integer_type(col_type: str) -> bool:
    t = col_type.upper()
    return "INT" in t


def _is_real_type(col_type: str) -> bool:
    t = col_type.upper()
    return ("REAL" in t) or ("FLOA" in t) or ("DOUB" in t)


def _is_blob_type(col_type: str) -> bool:
    return "BLOB" in col_type.upper()


def _is_numeric_type(col_type: str) -> bool:
    t = col_type.upper()
    return ("NUMERIC" in t) or ("DECIMAL" in t) or ("BOOLEAN" in t)


def _random_text(rng: random.Random, length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _random_blob(rng: random.Random, length: int = 12) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(length))


def _random_value_for_column(col: ColumnInfo, rng: random.Random):
    """
    Generate a random value based on declared column type.

    Note:
    - SQLite is dynamically typed, so this is best-effort generation.
    - Complex constraints (CHECK/UNIQUE/etc.) may still reject values.
    """
    ctype = col.col_type or ""
    if _is_text_like(ctype):
        return _random_text(rng, length=16)
    if _is_integer_type(ctype):
        return rng.randint(1, 1_000_000)
    if _is_real_type(ctype):
        return round(rng.uniform(0.01, 1_000_000.0), 6)
    if _is_blob_type(ctype):
        return _random_blob(rng, length=16)
    if _is_numeric_type(ctype):
        return round(rng.uniform(0.01, 10_000.0), 4)

    # For empty/unknown types, default to random text because SQLite accepts it.
    return _random_text(rng, length=16)


def _read_foreign_keys(
    conn: sqlite3.Connection,
    table_name: str,
) -> Dict[str, Dict[str, str]]:
    """
    Read single-column foreign key metadata via PRAGMA foreign_key_list.

    Returns:
    {
        "child_column_name": {
            "parent_table": "...",
            "parent_column": "..."
        },
        ...
    }

    Notes:
    - This handles single-column foreign keys directly.
    - Composite foreign keys are ignored here and will fall back to random generation.
    """
    rows = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
    if not rows:
        return {}

    grouped: Dict[int, List[sqlite3.Row]] = {}
    for row in rows:
        fk_id = int(row[0])
        grouped.setdefault(fk_id, []).append(row)

    result: Dict[str, Dict[str, str]] = {}
    for fk_id, fk_rows in grouped.items():
        if len(fk_rows) != 1:
            # Composite FK. Skip explicit handling in this version.
            continue

        row = fk_rows[0]
        child_col = str(row[3])
        parent_table = str(row[2])
        parent_col = str(row[4])

        result[child_col] = {
            "parent_table": parent_table,
            "parent_column": parent_col,
        }

    return result


def _sample_existing_fk_value(
    conn: sqlite3.Connection,
    parent_table: str,
    parent_column: str,
) -> object:
    """
    Sample one valid referenced value from the parent table.
    Raises if the parent table has no usable rows.
    """
    sql = (
        f'SELECT "{parent_column}" '
        f'FROM "{parent_table}" '
        f'WHERE "{parent_column}" IS NOT NULL '
        f'ORDER BY RANDOM() LIMIT 1'
    )
    row = conn.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(
            f"Cannot satisfy foreign key because parent table "
            f"'{parent_table}.{parent_column}' has no non-NULL rows."
        )
    return row[0]


def _build_insert_payload(
    columns: List[ColumnInfo],
    target_column: str,
    payload: str,
    rng: random.Random,
    conn: Optional[sqlite3.Connection] = None,
    table_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    Build a full row payload for INSERT.

    Rules:
    - Target column gets the provided payload.
    - Other non-PK columns get values.
    - Integer PK columns are omitted to let SQLite auto-assign rowid where possible.
    - Single-column FK fields get valid sampled referenced values when possible.
    """
    by_name = {c.name: c for c in columns}
    if target_column not in by_name:
        available = ", ".join(sorted(by_name))
        raise ValueError(
            f"Target column '{target_column}' not found. Available: {available}"
        )

    target_info = by_name[target_column]
    if not _is_text_like(target_info.col_type):
        raise ValueError(
            f"Target column '{target_column}' is not text-like. "
            f"Declared type: '{target_info.col_type}'"
        )

    fk_map: Dict[str, Dict[str, str]] = {}
    if conn is not None and table_name is not None:
        fk_map = _read_foreign_keys(conn, table_name)

    values: Dict[str, object] = {}
    for col in columns:
        # Skip integer primary key to let SQLite generate id automatically.
        if col.pk_position > 0 and _is_integer_type(col.col_type):
            continue

        if col.name == target_column:
            values[col.name] = payload
            continue

        # If this column is a single-column FK, use an existing referenced value.
        if col.name in fk_map and conn is not None:
            parent_table = fk_map[col.name]["parent_table"]
            parent_column = fk_map[col.name]["parent_column"]
            values[col.name] = _sample_existing_fk_value(
                conn, parent_table, parent_column
            )
            continue

        # Otherwise, use generated value.
        values[col.name] = _random_value_for_column(col, rng)

    return values


def _insert_row(
    conn: sqlite3.Connection,
    table_name: str,
    row_values: Dict[str, object],
) -> int:
    """
    Execute INSERT for the provided row dictionary.
    """
    if not row_values:
        raise ValueError("No values to insert.")

    cols = list(row_values.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{table_name}" ({col_sql}) VALUES ({placeholders})'
    params = [row_values[c] for c in cols]

    cur = conn.execute(sql, params)
    return int(cur.lastrowid)


def insert_payload_into_text_column(
    db_path: str | Path,
    table_name: str,
    target_column: str,
    payload: str,
    *,
    seed: int = 42,
    max_retries: int = 5,
) -> int:
    """
    Insert one row into `table_name`, placing `payload` in `target_column`.

    Other inserted columns are random values based on declared types,
    except single-column foreign keys, which are filled using existing
    referenced values from parent tables.

    Args:
    - db_path: path to SQLite file
    - table_name: target table
    - target_column: column that must receive payload (must be text-like)
    - payload: payload string to inject
    - seed: seed for deterministic random generation
    - max_retries: retries in case generated row violates constraints

    Returns:
    - inserted row id (lastrowid)
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB file not found: {db_path}")

    rng = random.Random(seed)
    with sqlite3.connect(db_path) as conn:
        # Make FK checks explicit and deterministic.
        conn.execute("PRAGMA foreign_keys = ON")

        columns = _read_table_schema(conn, table_name)

        last_error: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                row_values = _build_insert_payload(
                    columns=columns,
                    target_column=target_column,
                    payload=payload,
                    rng=rng,
                    conn=conn,
                    table_name=table_name,
                )
                row_id = _insert_row(conn, table_name, row_values)
                conn.commit()
                return row_id
            except sqlite3.IntegrityError as e:
                # Retry with fresh random values; UNIQUE/CHECK/etc. may reject a sample.
                last_error = e
                conn.rollback()
                continue
            except RuntimeError as e:
                # Parent FK table may be empty, or sampled FK could not be satisfied.
                last_error = e
                conn.rollback()
                continue

        raise RuntimeError(
            f"Failed to insert into '{table_name}.{target_column}' after "
            f"{max_retries} retries due to constraints. "
            f"Last error: {last_error}"
        )


def insert_payload_into_text_columns(
    db_path: str | Path,
    table_name: str,
    column_names: Sequence[str],
    payload: str,
    *,
    seed: int = 42,
    max_retries: int = 5,
) -> Dict[str, int]:
    """
    Insert one row per target column (for all text-like columns in `column_names`).

    For each column in `column_names`:
    - verify column exists and is text-like
    - create a new row with payload in that column
    - fill other columns with generated values
    - use valid referenced values for single-column foreign keys

    Returns:
    - mapping: {column_name: inserted_row_id}
    """
    if not column_names:
        raise ValueError("column_names must not be empty.")

    result: Dict[str, int] = {}
    for idx, col_name in enumerate(column_names):
        # Offset seed for deterministic but distinct rows across columns.
        row_id = insert_payload_into_text_column(
            db_path=db_path,
            table_name=table_name,
            target_column=col_name,
            payload=payload,
            seed=seed + idx,
            max_retries=max_retries,
        )
        result[col_name] = row_id
    return result

