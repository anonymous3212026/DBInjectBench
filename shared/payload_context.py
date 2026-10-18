from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


def count_context_value(obj: Any, value: str) -> int:
    """Count occurrences of a value anywhere in a nested context pack."""
    if not value:
        return 0
    if isinstance(obj, str):
        return obj.count(value)
    if isinstance(obj, list):
        return sum(count_context_value(item, value) for item in obj)
    if isinstance(obj, dict):
        return sum(count_context_value(item, value) for item in obj.values())
    return 0


@dataclass(frozen=True)
class PayloadRetrievalStatus:
    payload_forced: bool
    natural_payload_occurrences: int
    table: str | None = None
    column: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload_forced": self.payload_forced,
            "payload_retrieved_naturally": not self.payload_forced,
            "natural_payload_occurrences": self.natural_payload_occurrences,
            "payload_forced_table": self.table,
            "payload_forced_column": self.column,
        }


def ensure_payload_in_context(
    context_pack: Dict[str, Any],
    *,
    injected_value: str,
    table_to_columns: Mapping[str, List[str]],
) -> PayloadRetrievalStatus:
    """Assume retrieval of an injected value while preserving natural results.

    If retrieval already returned the value, the context is left untouched.
    Otherwise, append one normal-looking value-candidate entry for the first
    configured injection target. This keeps the assumed-retrieval context small
    and makes the intervention explicit in the returned metadata.
    """
    natural_occurrences = count_context_value(context_pack, injected_value)
    if natural_occurrences > 0:
        return PayloadRetrievalStatus(
            payload_forced=False,
            natural_payload_occurrences=natural_occurrences,
        )

    target = next(
        (
            (str(table), str(column))
            for table, columns in table_to_columns.items()
            for column in columns
        ),
        None,
    )
    if target is None:
        raise ValueError("Cannot force payload retrieval without an injected table/column target.")

    table, column = target
    candidates = context_pack.setdefault("value_candidates", [])
    if not isinstance(candidates, list):
        raise TypeError("context_pack['value_candidates'] must be a list.")
    candidates.append({"table": table, "column": column, "values": [injected_value]})

    return PayloadRetrievalStatus(
        payload_forced=True,
        natural_payload_occurrences=0,
        table=table,
        column=column,
    )
