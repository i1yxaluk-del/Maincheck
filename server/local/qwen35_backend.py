from __future__ import annotations

import asyncio
import os


class Qwen35Backend:
    """Lazy text-only Transformers backend used by experimental v3.1 X/Y.

    The class name is retained for API compatibility, but v3.1 no longer loads
    the multimodal Qwen3.5 processor. The default model is a text-only Russian
    GEC checkpoint, so Pillow/Torchvision are not required.
    """

    MODEL_ID = os.getenv(
        "GEC_SPECIALIST_MODEL",
        "ReginaNasyrova/checkpoint_150_lora_grpo_upd_reward_GECExplanation-4B-sft-stage1-March2026",
    )
    _shared_instance: "Qwen35Backend | None" = None

    def __new__(cls):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
            cls._shared_instance._initialized = False
        return cls._shared_instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._tokenizer = None
        self._model = None
        self._device = None
        self.max_new = int(os.getenv("GEC_SPECIALIST_MAX_NEW_TOKENS", "512"))
        self._initialized = True

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )
        self._device = next(self._model.parameters()).device

    async def correct(self, text: str, context: str, temperature: float = 0.10) -> str:
        return await asyncio.to_thread(self._correct_sync, text, context, temperature)

    def _correct_sync(self, text: str, context: str, temperature: float) -> str:
        import torch

        self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — специализированный корректор русского текста. "
                    "Исправляй грамматические, орфографические и пунктуационные ошибки. "
                    "Сохраняй смысл, термины, порядок слов и структуру абзацев. "
                    "Не переписывай текст ради стиля. Верни только исправленный текст."
                ),
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context[-3500:]}\n\nИсходный текст:\n{text}",
            },
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            kwargs = {
                **inputs,
                "max_new_tokens": self.max_new,
                "repetition_penalty": 1.05,
            }
            if temperature > 0:
                kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.85})
            else:
                kwargs.update({"do_sample": False})
            output = self._model.generate(**kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    async def warmup(self) -> None:
        await self.correct("Проверка запуска.", "Проверка запуска.", temperature=0.0)
