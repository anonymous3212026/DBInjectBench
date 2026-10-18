from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dotenv import load_dotenv

from .agent_planner import PlannerAgent
from .agent_retriever import RetrieverAgent
from .agent_sql_generator import SQLGeneratorAgent
from .llm import LocalChatLLM
from .sqlite_tools import SQLiteTools
from shared.payload_context import ensure_payload_in_context


load_dotenv()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _sanitize_for_path(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "unknown_model"


def _infer_llm_name(model_name: str) -> str:
    lower = model_name.lower()
    for family in ("llama", "qwen", "mistral", "gemma", "phi", "deepseek"):
        if family in lower:
            return family
    return _sanitize_for_path(model_name)


_BASE_LLM = None


def _get_llm():
    global _BASE_LLM
    if _BASE_LLM is None:
        _BASE_LLM = LocalChatLLM()
    return _BASE_LLM


class TraceableLLM:
    def __init__(self, base_llm: Optional[LocalChatLLM] = None):
        self._base = base_llm or _get_llm()
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages: Sequence[Dict[str, str]]) -> str:
        msg_copy = deepcopy(list(messages))
        output = self._base.chat(list(messages))
        self.calls.append({"messages": msg_copy, "output": output})
        return output

    def mark(self) -> int:
        return len(self.calls)

    def calls_since(self, marker: int) -> List[Dict[str, Any]]:
        return self.calls[marker:]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _append_heading(lines: List[str], title: str) -> None:
    bar = "=" * 90
    lines.append(bar)
    lines.append(title)
    lines.append(bar)
    lines.append("")


def _append_step(
    lines: List[str],
    title: str,
    step_input: Dict[str, Any],
    step_output: Dict[str, Any],
    llm_calls: List[Dict[str, Any]],
) -> None:
    _append_heading(lines, title)
    lines.append("Input:")
    lines.append(_pretty_json(step_input))
    lines.append("")

    lines.append("Output:")
    lines.append(_pretty_json(step_output))
    lines.append("")

    lines.append("LLM Calls:")
    if not llm_calls:
        lines.append("  (none)")
        lines.append("")
        return

    for i, call in enumerate(llm_calls, start=1):
        lines.append("-" * 90)
        lines.append(f"LLM Call {i} Input:")
        lines.append(_pretty_json({"messages": call.get("messages", [])}))
        lines.append("")
        lines.append(f"LLM Call {i} Output:")
        lines.append(_pretty_json({"output": call.get("output", "")}))
        lines.append("")


def pipeline(
    question: str,
    db_path: str,
    model_name: str,
    question_id: Optional[str] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    enable_logging: bool = True,
    injected_value: Optional[str] = None,
    injected_table_to_columns: Optional[Mapping[str, List[str]]] = None,
    retrieval_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    q = (question or "").strip()
    if not q:
        raise ValueError("question must be a non-empty string")

    db_path = (db_path or "").strip()
    if not db_path:
        raise ValueError("db_path must be a non-empty string")

    model_name = (model_name or "").strip()
    if not model_name:
        raise ValueError("model_name must be a non-empty string")

    llm_name = _sanitize_for_path(_infer_llm_name(model_name))
    logs_root = os.getenv("PIPELINE_LOGS_ROOT", "").strip() or "logs"

    os.environ["MODEL_NAME"] = model_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qid = str(question_id or f"single_{timestamp}")

    ex: Dict[str, Any] = {"question_id": qid, "question": q}
    if extra_payload:
        ex.update(extra_payload)
        ex["question_id"] = qid
        ex["question"] = q

    run_root: Optional[Path] = None
    q_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    out_path: Optional[Path] = None

    if enable_logging:
        run_root = Path(logs_root) / llm_name / timestamp
        run_root.mkdir(parents=True, exist_ok=True)

        q_dir = run_root / qid
        q_dir.mkdir(parents=True, exist_ok=True)
        log_path = q_dir / "log.txt"
        out_path = q_dir / "output.txt"

    llm = TraceableLLM()
    tools = SQLiteTools(db_path)
    planner = PlannerAgent(llm, db_path)
    retriever = RetrieverAgent(llm, tools)
    sqlgen = SQLGeneratorAgent(llm, tools)

    summary: List[Dict[str, Any]] = []
    pred_sql = ""
    failed = False
    error_repr = ""
    tb_text = ""

    log_lines: List[str] = []
    if enable_logging:
        _append_heading(log_lines, "QUESTION RUN")
        log_lines.append(f"question_id: {qid}")
        log_lines.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
        log_lines.append(f"model_name: {model_name}")
        log_lines.append(f"llm_name: {llm_name}")
        log_lines.append(f"db_path: {db_path}")
        log_lines.append("")
        log_lines.append("Question Payload:")
        log_lines.append(_pretty_json(ex))
        log_lines.append("")

    try:
        marker = llm.mark()
        planner_input = {"question": q}
        plan = planner.run(q)
        planner_calls = llm.calls_since(marker)
        if enable_logging:
            _append_step(log_lines, "Planner Step", planner_input, plan, planner_calls)

        marker = llm.mark()
        retriever_input = {"question": q, "plan": plan}
        context_pack = retriever.run(q, plan)
        if injected_value is not None:
            retrieval = ensure_payload_in_context(
                context_pack,
                injected_value=injected_value,
                table_to_columns=injected_table_to_columns or {},
            )
            if retrieval_metadata is not None:
                retrieval_metadata.update(retrieval.to_dict())
        retriever_calls = llm.calls_since(marker)
        if enable_logging:
            _append_step(
                log_lines,
                "Retriever Step",
                retriever_input,
                context_pack,
                retriever_calls,
            )

        marker = llm.mark()
        sqlgen_input = {"question": q, "context_pack": context_pack}
        sql_out = sqlgen.run(q, context_pack)
        sqlgen_calls = llm.calls_since(marker)
        if enable_logging:
            _append_step(
                log_lines,
                "SQL Generator Step",
                sqlgen_input,
                sql_out,
                sqlgen_calls,
            )

        pred_sql = str(sql_out.get("final_sql") or "").strip()
        if not pred_sql:
            failed = True
            error_repr = repr(RuntimeError("final_sql is empty"))
            tb_text = "final_sql is empty"

    except Exception as exc:
        failed = True
        error_repr = repr(exc)
        tb_text = traceback.format_exc()
        if enable_logging:
            _append_heading(log_lines, "PIPELINE ERROR")
            log_lines.append(f"error: {repr(exc)}")
            log_lines.append("")
            log_lines.append("traceback:")
            log_lines.append(tb_text)
            log_lines.append("")

    finally:
        tools.close()

    output_obj = dict(ex)
    output_obj["pred_sql"] = pred_sql
    if retrieval_metadata:
        output_obj.update(retrieval_metadata)
    if failed and error_repr:
        output_obj["error"] = error_repr

    if enable_logging and log_path is not None and out_path is not None and run_root is not None:
        _write_text(log_path, "\n".join(log_lines).rstrip() + "\n")
        _write_json(out_path, output_obj)

        summary.append(
            {
                "question_id": qid,
                "failed": failed,
                "pred_sql": pred_sql,
                "log_path": str(log_path),
                "output_path": str(out_path),
            }
        )

        _write_json(
            run_root / "run_summary.json",
            {
                "model_name": model_name,
                "llm_name": llm_name,
                "db_path": db_path,
                "questions_path": "__pipeline_callable_single_question__",
                "timestamp": timestamp,
                "num_questions": 1,
                "results": summary,
            },
        )

    if failed:
        if enable_logging and q_dir is not None:
            raise RuntimeError(
                f"Pipeline failed for question_id={qid}. See logs under {q_dir}\n{tb_text or 'pipeline failed'}"
            )
        raise RuntimeError(
            f"Pipeline failed for question_id={qid}\n{tb_text or 'pipeline failed'}"
        )

    return pred_sql


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run pipeline for one question and return SQL."
    )
    p.add_argument("question", help="Natural language question")
    p.add_argument(
        "--db-path",
        required=True,
        help="Path to the SQLite database",
    )
    p.add_argument(
        "--model-name",
        required=True,
        help="Model name to use",
    )
    p.add_argument(
        "--question-id",
        default="",
        help="Optional question id for log folder naming.",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Disable all logging and file outputs.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    sql = pipeline(
        question=args.question,
        db_path=args.db_path,
        model_name=args.model_name,
        question_id=args.question_id or None,
        enable_logging=not args.no_log,
    )
    print(sql)


if __name__ == "__main__":
    main()
