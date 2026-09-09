from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("ai_suggester.russian_gec")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_ADAPTER_REPO = "synterr-nlp/bea2026-gec-adapters"
DEFAULT_ADAPTER_SUBFOLDER = "v4_qwen35_08b_lorugec"

SYSTEM_PROMPT = (
    "Исправь в русском тексте объективные ошибки грамматики, управления, "
    "согласования, форм слов, орфографии и пунктуации. "
    "Сохрани смысл, слова, термины, числа и структуру текста. "
    "Ничего не объясняй и не переписывай текст ради стиля. "
    "Верни только исправленный текст."
)


@dataclass(frozen=True)
class GecStats:
    enabled: bool
    loaded: bool
    base_model: str
    adapter: str
    calls: int


class RussianGecSpecialist:
    """Qwen3.5-0.8B + BEA2026 SyntErr LoRA as a local GEC candidate generator."""

    def __init__(self) -> None:
        self.enabled = os.getenv("RUSSIAN_GEC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        self.base_model = os.getenv("RUSSIAN_GEC_BASE_MODEL", DEFAULT_BASE_MODEL)
        self.adapter_repo = os.getenv("RUSSIAN_GEC_ADAPTER_REPO", DEFAULT_ADAPTER_REPO)
        self.adapter_subfolder = os.getenv("RUSSIAN_GEC_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER)
        self.max_new_tokens = int(os.getenv("RUSSIAN_GEC_MAX_NEW_TOKENS", "256"))
        self.max_input_tokens = int(os.getenv("RUSSIAN_GEC_MAX_INPUT_TOKENS", "768"))
        self.threads = int(os.getenv("RUSSIAN_GEC_THREADS", "4"))
        self._tokenizer = None
        self._model = None
        self._calls = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return self.enabled

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            torch.set_num_threads(max(1, self.threads))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        logger.info(
            "Loading Russian GEC specialist: base=%s adapter=%s/%s",
            self.base_model,
            self.adapter_repo,
            self.adapter_subfolder,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype="auto",
        )
        self._model = PeftModel.from_pretrained(
            base,
            self.adapter_repo,
            subfolder=self.adapter_subfolder,
        )
        self._model.eval()

    async def correct(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return text
        self._calls += 1
        async with self._lock:
            return await asyncio.to_thread(self._correct_sync, text)

    def _correct_sync(self, text: str) -> str:
        import torch

        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.02,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    async def warmup(self) -> None:
        if self.enabled:
            await self.correct("Проверка запуска корректора.")

    def metrics(self) -> GecStats:
        return GecStats(
            enabled=self.enabled,
            loaded=self._model is not None,
            base_model=self.base_model,
            adapter=f"{self.adapter_repo}:{self.adapter_subfolder}",
            calls=self._calls,
        )
