#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def force_payload_str(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def create_temp_database_copy(source_db_path: Path) -> Path:
    (fd, tmp_path) = tempfile.mkstemp(prefix='autoinject_tmp_', suffix='.sqlite3')
    os.close(fd)
    tmp_db_path = Path(tmp_path)
    shutil.copy2(source_db_path, tmp_db_path)
    return tmp_db_path


def cleanup_temp_database(temp_db_path: Path) -> None:
    try:
        if temp_db_path.exists():
            temp_db_path.unlink()
    except Exception:
        pass


def parse_relevant_columns(relevant_columns: List[str]) -> Dict[str, List[str]]:
    table_to_columns: Dict[str, List[str]] = {}
    for raw_name in relevant_columns:
        item = raw_name.strip()
        if not item:
            continue
        parts = item.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid relevant column '{item}'. Expected format 'table.column'.")
        (table_name, column_name) = (parts[0].strip(), parts[1].strip())
        if not table_name or not column_name:
            raise ValueError(f"Invalid relevant column '{item}'. Empty table or column.")
        table_to_columns.setdefault(table_name, [])
        if column_name not in table_to_columns[table_name]:
            table_to_columns[table_name].append(column_name)
    return table_to_columns


def run_victim_pipeline(question: str, db_path: Path, victim_model_name: str, *, injected_value: Optional[str]=None, injected_table_to_columns: Optional[Dict[str, List[str]]]=None, retrieval_metadata: Optional[Dict[str, Any]]=None) -> str:
    from nl2sql.pipeline_callable import pipeline
    result = pipeline(question, str(db_path), victim_model_name, injected_value=injected_value, injected_table_to_columns=injected_table_to_columns, retrieval_metadata=retrieval_metadata)
    if not isinstance(result, str):
        raise TypeError(f'Victim pipeline must return SQL string, got {type(result)}')
    return result


def run_single_inference(item: Dict[str, Any]) -> Dict[str, Any]:
    q_idx = item.get("idx")
    db_name = force_payload_str(item.get("db_id")).strip()
    question = force_payload_str(item.get("question"))
    relevant_columns_raw = item.get("relevant_columns", [])
    relevant_columns = [force_payload_str(x) for x in relevant_columns_raw]

    final_mal_content = force_payload_str(
        item.get("final_string_to_add_in_database")
    )
    if not final_mal_content:
        final_mal_content = force_payload_str(item.get("final_string_to_add"))

    victim_model_name = force_payload_str(item.get("victim_model_name")).strip()

    if q_idx is None:
        raise ValueError("Missing q_idx")
    if not db_name:
        raise ValueError(f"q_id={q_idx}: missing db_id")
    if not question:
        raise ValueError(f"q_id={q_idx}: missing question")
    if not relevant_columns:
        raise ValueError(f"q_id={q_idx}: missing relevant_columns")
    if not final_mal_content:
        raise ValueError(
            f"q_id={q_idx}: missing final_string_to_add_in_database"
        )
    if not victim_model_name:
        raise ValueError(f"q_id={q_idx}: missing victim_model_name")

    db_path = Path(os.getenv("ARTIFACT_DATABASE_DIR", "database")) / f"{db_name}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"q_id={q_idx}: database not found: {db_path}")

    table_to_columns = parse_relevant_columns(relevant_columns)

    temp_db_path: Optional[Path] = None
    predicted_sql = ""
    error: Optional[str] = None
    retrieval_metadata: Dict[str, Any] = {
        "payload_forced": False,
        "payload_retrieved_naturally": False,
        "natural_payload_occurrences": 0,
        "payload_forced_table": None,
        "payload_forced_column": None,
    }

    try:
        from shared.db_payload_inserter import insert_payload_into_text_columns

        temp_db_path = create_temp_database_copy(db_path)

        for idx, (table_name, column_names) in enumerate(table_to_columns.items()):
            insert_payload_into_text_columns(
                db_path=temp_db_path,
                table_name=table_name,
                column_names=column_names,
                payload=final_mal_content,
                seed=42 + idx,
                max_retries=5,
            )

        predicted_sql = run_victim_pipeline(
            question=question,
            db_path=temp_db_path,
            victim_model_name=victim_model_name,
            injected_value=final_mal_content,
            injected_table_to_columns=table_to_columns,
            retrieval_metadata=retrieval_metadata,
        )

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    finally:
        if temp_db_path is not None:
            cleanup_temp_database(temp_db_path)

    result = dict(item)
    result["final_string_to_add_in_database"] = final_mal_content
    result["pipeline_output_sql"] = predicted_sql
    result["status"] = "failed" if error else "success"
    result["error"] = error
    result.update(retrieval_metadata)
    result["time_utc"] = datetime.now(timezone.utc).isoformat()

    return result


def evaluate_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    from evaluation.classify_sql import classify_sql_case

    q_idx = item.get("idx")
    gold_sql = item.get("SQL", "")
    target_sql = item.get("target_sql", "")
    generated_sql = item.get("pipeline_output_sql", "")
    target_attack_type = item.get("attack_category", "")

    if q_idx is None:
        raise ValueError("Missing idx")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError("Missing or invalid SQL")
    if not isinstance(target_sql, str) or not target_sql.strip():
        raise ValueError("Missing or invalid target_sql")
    if not isinstance(generated_sql, str):
        raise ValueError("Invalid pipeline_output_sql")
    if not isinstance(target_attack_type, str) or not target_attack_type.strip():
        raise ValueError("Missing or invalid attack_category")

    eval_result = classify_sql_case(
        gold_sql=gold_sql,
        target_sql=target_sql,
        generated_sql=generated_sql,
        target_attack_type=target_attack_type,
    )

    output = dict(item)
    output["eval"] = eval_result
    output["eval_status"] = "success"
    output["eval_error"] = None
    output["eval_time_utc"] = datetime.now(timezone.utc).isoformat()

    return output


def run_inference_and_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    inference_result = run_single_inference(data)

    try:
        return evaluate_single_item(inference_result)
    except Exception:
        output = dict(inference_result)
        output["eval"] = None
        output["eval_status"] = "failed"
        output["eval_error"] = traceback.format_exc()
        output["eval_time_utc"] = datetime.now(timezone.utc).isoformat()
        return output


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)




