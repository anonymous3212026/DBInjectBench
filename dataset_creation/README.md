# Dataset Creation

Start with raw question data in `raw_data/superhero.json` or `raw_data/financial.json`. Each JSON record must contain `question_id`, `db_id`, `question`, and `SQL`; `difficulty` and `evidence` are optional. Supply the matching database in `database/` and set `OPENAI_API_KEY` in `.env`. Raw question files and SQLite databases are supplied separately. The repository's `data/` folder contains only ready-to-use trainer inputs.

Run from the repository root:

```bash
DATASET=superhero

# 1. Identify relevant columns, retain text columns, and remove empty cases.
python -m dataset_creation.relevant_column_identifier \
  --input "raw_data/${DATASET}.json" \
  --db "database/${DATASET}.sqlite" \
  --output "outputs/${DATASET}/columns.json"

# 2. Generate and validate target SQL for the four attack categories.
python -m dataset_creation.target_sql_generator \
  --dataset "$DATASET" \
  --questions "outputs/${DATASET}/columns.json" \
  --db "database/${DATASET}.sqlite" \
  --output "outputs/${DATASET}/target_sql.json"

# 3. Generate and validate base attack payloads.
python -m dataset_creation.base_attack_payload_generator \
  --dataset "$DATASET" \
  --input "outputs/${DATASET}/target_sql.json" \
  --db "database/${DATASET}.sqlite" \
  --output "outputs/${DATASET}/dataset.json"
```

Use `DATASET=financial` for Financial. The final data is saved to `outputs/superhero/dataset.json` or `outputs/financial/dataset.json`, ready to pass to the GRPO runner with `--input`.

Default models are `gpt-4o`, `gpt-6-luna`, and `gpt-6-sol` for the three stages respectively. Override them with `--model`. Dataset-specific prompt instructions and examples are in `configs/`. The two generation stages validate candidates, retry up to three times by default, save unresolved failures separately, and resume accepted records on rerun. Use each script's `--help` for its available options.
