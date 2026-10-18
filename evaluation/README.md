# Evaluation

`classify_sql.py` provides `classify_sql_case`; `prompts.py` contains the judge prompt. The API uses `GEMINI_API_KEY` from the environment or the repository-root `.env` file. The default judge is `gemini-2.5-flash`.

From the repository root:

```bash
python -m evaluation.run --input outputs/qwen_example --output outputs/qwen_judge.json
```

The input can be one saved trainer `result.json`, a directory containing such files, or a JSON object/list with `SQL`/`gold_sql`, `target_sql`, `generated_sql`/`pipeline_output_sql`, and `attack_category`/`target_attack_type`. Use `--validate-only` to check inputs without API calls; `--model` selects a different judge model.

Categories are 1 (no attack), 2 (wrong attack type), 3 (correct attack type, wrong behavior), and 4 (target achieved). The output retains each scored case, API/input failures, category counts, and separate rates for any attack (2–4) and target achievement (4). Rate denominators exclude failed/unscored cases, whose count remains explicit.

`inference.py` contains the temporary-database and local-pipeline inference helpers used by training. Final model-specific inference and baseline evaluation are performed by the selected trainer. Scoring saved predictions through `run.py` needs no GPU and makes no additional victim-model calls.
