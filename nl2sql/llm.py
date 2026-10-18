import os
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class LocalChatLLM:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip() or None

        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2056"))
        if "ministral-3" in self.model_name.lower():
            from transformers import (
                Mistral3ForConditionalGeneration,
                MistralCommonBackend,
            )

            self.tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model_family = "mistral3"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model_family = "causal_lm"

    @staticmethod
    def _normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "user", "content": ""}]

        # 1) keep at most one leading system message
        out: List[Dict[str, str]] = []
        i = 0
        if messages[0].get("role") == "system":
            out.append({"role": "system", "content": messages[0].get("content", "")})
            i = 1

        # 2) merge consecutive same-role messages (user+user, assistant+assistant)
        merged: List[Dict[str, str]] = []
        for m in messages[i:]:
            role = (m.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = m.get("content", "")
            if not merged:
                merged.append({"role": role, "content": content})
                continue
            if merged[-1]["role"] == role:
                prev = merged[-1]["content"]
                merged[-1]["content"] = (prev + "\n" + content).strip()
            else:
                merged.append({"role": role, "content": content})

        # 3) enforce alternation after system: must start with user
        if merged and merged[0]["role"] == "assistant":
            merged.insert(0, {"role": "user", "content": ""})

        fixed: List[Dict[str, str]] = []
        expected = "user"
        for m in merged:
            role = m["role"]
            content = m["content"]

            if role != expected:
                # if it is the wrong role, fold content into the last message (same role)
                if fixed:
                    fixed[-1]["content"] = (fixed[-1]["content"] + "\n" + content).strip()
                else:
                    fixed.append({"role": expected, "content": content})
            else:
                fixed.append({"role": role, "content": content})
                expected = "assistant" if expected == "user" else "user"

        out.extend(fixed)
        return out

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]]) -> str:
        msgs = self._normalize_chat_messages(list(messages))

        if self.model_family == "mistral3":
            tokenized = self.tokenizer.apply_chat_template(
                msgs,
                return_tensors="pt",
                return_dict=True,
            )
            #tokenized["input_ids"] = tokenized["input_ids"].to(self.model.device)
            tokenized = {
                            k: v.to(self.model.device) if hasattr(v, "to") else v
                            for k, v in tokenized.items()
                        }

            output_ids = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
            )[0]

            return self.tokenizer.decode(
                output_ids[len(tokenized["input_ids"][0]):],
                skip_special_tokens=True,
            ).strip()

        else:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
            )
            gen = output_ids[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
