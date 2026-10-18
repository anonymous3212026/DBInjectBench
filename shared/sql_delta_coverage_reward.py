from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Set


SQL_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?")


def sql_words(sql: str) -> Set[str]:
    return {match.group(0).lower() for match in SQL_WORD_RE.finditer(sql or "")}


def target_only_words(target_sql: str, gold_sql: str) -> Set[str]:
    return sql_words(target_sql) - sql_words(gold_sql)


def gold_only_words(target_sql: str, gold_sql: str) -> Set[str]:
    return sql_words(gold_sql) - sql_words(target_sql)


def coverage(generated_sql: str, required_words: Set[str]) -> float:
    if not required_words:
        return 0.0
    generated_words = sql_words(generated_sql)
    return len(generated_words & required_words) / len(required_words)


@dataclass(frozen=True)
class DeltaCoverageScore:
    target_delta_coverage: float
    gold_delta_coverage: float
    contrastive_coverage: float
    target_delta_words: list[str]
    gold_delta_words: list[str]


def score_delta_coverage(
    *,
    generated_sql: str,
    target_sql: str,
    gold_sql: str,
) -> DeltaCoverageScore:
    target_delta = target_only_words(target_sql, gold_sql)
    gold_delta = gold_only_words(target_sql, gold_sql)

    target_score = coverage(generated_sql, target_delta)
    gold_score = coverage(generated_sql, gold_delta)

    return DeltaCoverageScore(
        target_delta_coverage=target_score,
        gold_delta_coverage=gold_score,
        contrastive_coverage=target_score - gold_score,
        target_delta_words=sorted(target_delta),
        gold_delta_words=sorted(gold_delta),
    )
