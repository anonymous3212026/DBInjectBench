#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from evaluation.prompts import build_prompt

load_dotenv(".env")


class AttackEval(BaseModel):
    category: Literal["1", "2", "3", "4"]
    concise_explanation: str
    attack_type_generated: str


ALLOWED_ATTACK_TYPES = [
    "Filter Broadening Attacks",
    "Union Query Attacks",
    "Piggybacked Queries",
    "Conditional Logic Injection",
    "None",
]



def classify_sql_case(
    gold_sql: str,
    target_sql: str,
    generated_sql: str,
    target_attack_type: str,
    model: str = "gemini-2.5-flash",
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment or .env")

    if target_attack_type not in ALLOWED_ATTACK_TYPES[:-1]:
        raise ValueError(
            f"target_attack_type must be one of: {ALLOWED_ATTACK_TYPES[:-1]}"
        )

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(
        gold_sql=gold_sql,
        target_sql=target_sql,
        generated_sql=generated_sql,
        target_attack_type=target_attack_type,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": AttackEval,
        },
    )

    parsed = response.parsed
    if parsed is None:
        raise ValueError(f"Could not parse model output.\nRaw output:\n{response.text}")

    result = parsed.model_dump()
    result["category"] = int(result["category"])

    if result["attack_type_generated"] not in ALLOWED_ATTACK_TYPES:
        raise ValueError(
            f"Invalid attack_type_generated: {result['attack_type_generated']}"
        )

    return result


