from __future__ import annotations

import asyncio
import os


class Qwen35Backend:
    """Lazy Transformers backend for official Qwen3.5-4B.

    It is experimental only. The model is loaded inside the existing uvicorn
    process; no second HTTP/uvicorn service is started.
    """

    MODEL_ID = os.getenv("QWEN35_MODEL", "Qwen/Qwen3.5-4B")

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None
        self.max_new = int(os.getenv("QWEN35_MAX_NEW_TOKENS", "768"))

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )
        self._device = next(self._model.parameters()).device

    async def correct(self, text: str, context: str, temperature: float = 0.7) -> str:
        return await asyncio.to_thread(self._correct_sync, text, context, temperature)

    def _correct_sync(self, text: str, context: str, temperature: float) -> str:
        import torch
        self._load()
        messages = [
            {"role": "system", "content": "Ты профессиональный корректор русского официально-делового текста. Исправляй только объективные ошибки. Сохраняй смысл, порядок слов, абзацы и термины. Верни только исправленный текст без пояснений."},
            {"role": "user", "content": f"КОНТЕКСТ:\n{context[-3000:]}\n\nТЕКСТ:\n{text}"},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return self._processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()

    async def warmup(self) -> None:
        await self.correct("Проверка запуска.", "Проверка запуска.", temperature=0.0)
