import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .utils import is_read_only_sql


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SQLiteTools:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def list_tables(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return [r[0] for r in cur.fetchall()]

    def describe_table(self, table: str) -> List[Dict[str, Any]]:
        t = _quote_ident(table)
        cur = self.conn.execute(f"PRAGMA table_info({t});")
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            out.append(
                {
                    "cid": r["cid"],
                    "name": r["name"],
                    "type": r["type"],
                    "notnull": bool(r["notnull"]),
                    "pk": int(r["pk"]),
                }
            )
        return out

    def foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        t = _quote_ident(table)
        cur = self.conn.execute(f"PRAGMA foreign_key_list({t});")
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            out.append(
                {
                    "ref_table": r["table"],
                    "from_col": r["from"],
                    "to_col": r["to"],
                }
            )
        return out

    def sample_distinct(self, table: str, column: str, k: int = 500) -> List[Any]:
        t = _quote_ident(table)
        c = _quote_ident(column)
        cur = self.conn.execute(
            f"SELECT DISTINCT {c} AS v FROM {t} WHERE {c} IS NOT NULL ORDER BY RANDOM() LIMIT {int(k)};"
        )
        return [r["v"] for r in cur.fetchall()]

    def search_like(self, table: str, column: str, pattern: str, k: int = 500) -> List[Any]:
        t = _quote_ident(table)
        c = _quote_ident(column)
        cur = self.conn.execute(
            f"SELECT DISTINCT {c} AS v FROM {t} "
            f"WHERE {c} IS NOT NULL AND {c} LIKE ? ORDER BY RANDOM() LIMIT {int(k)};",
            (pattern,),
        )
        return [r["v"] for r in cur.fetchall()]

    def run_sql(self, sql: str, max_rows: int) -> Dict[str, Any]:
        if not is_read_only_sql(sql):
            return {"ok": False, "error": "Only SELECT or WITH is allowed"}

        try:
            cur = self.conn.execute(sql)
            rows = cur.fetchmany(int(max_rows))
            cols = [d[0] for d in (cur.description or [])]
            return {"ok": True, "columns": cols, "rows": [list(r) for r in rows]}
        except Exception as e:
            return {"ok": False, "error": str(e)}
