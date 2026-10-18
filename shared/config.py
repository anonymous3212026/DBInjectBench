"""Configuration shared by the portable training entry point."""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path


def load_training_config(config_class):
    path = os.getenv("ARTIFACT_TRAINING_CONFIG")
    if not path:
        return config_class()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(config_class)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    return config_class(**values)
