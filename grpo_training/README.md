# GRPO Training

Run from the repository root through the shared CLI:

```bash
python -m grpo_training.run --model qwen --input data/superhero.json --idx 0 --output-dir outputs/qwen_example
```

Choose `llama`, `qwen`, `secalign`, `gpt`, or `gemini`. These select the corresponding `.py` implementation in this directory; the model name describes the victim, not the attacker. The default attacker checkpoint is defined by `AttackConfig` in `shared/attacker_core.py`.

`--help` lists selection and path options. `--validate-only` checks records and database files without loading models. `--config configs/<model>.json` (use the full repository-relative path `grpo_training/configs/<model>.json`) supplies training overrides; partial JSON objects are allowed. Unknown keys are rejected.

Each example runs in an isolated process. Outputs are recorded under the requested directory, with one subdirectory per case. All five trainers default to 100 policy update steps and 80 payload tokens; these values are mirrored in the JSON configuration files. Llama, Qwen, and SecAlign use 25% category-aligned coverage and 75% target-query probability. GPT and Gemini retain their coverage-only reward. The root README describes the baseline gate, forced-retrieval fallback, model requirements, and output conventions.

GPT defaults to `gpt-6-luna`; Gemini defaults to `gemini-3.5-flash-lite`. All five trainers use categories 2–4 as attack success for training ASR and category-based early stopping. Reward weights and failure handling are unchanged.
