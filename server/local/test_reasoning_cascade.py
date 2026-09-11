from __future__ import annotations

import asyncio

from reasoning_cascade import ReasoningCascade


def test_reasoner_is_followed_by_extractor_and_diff_is_bounded():
    cascade = ReasoningCascade()
    calls: list[str] = []

    async def fake_chat(model, messages, timeout, num_predict):
        calls.append(model)
        if len(calls) == 1:
            return "<think>После 'ночных' нужно множественное число.</think>\nИСПРАВЛЕННЫЙ ТЕКСТ: После ночных нарядов."
        return "После ночных нарядов."

    cascade._chat = fake_chat  # type: ignore[method-assign]
    edits = asyncio.run(cascade.candidates("После ночных наряда."))
    assert calls == [cascade.reasoner, cascade.extractor]
    assert [(e.before, e.after) for e in edits] == [("наряда", "нарядов")]
    assert all(e.category == "russian-gec-reasoning" for e in edits)


def test_empty_extractor_output_fails_closed():
    cascade = ReasoningCascade()
    calls = 0

    async def fake_chat(model, messages, timeout, num_predict):
        nonlocal calls
        calls += 1
        return "размышление" if calls == 1 else ""

    cascade._chat = fake_chat  # type: ignore[method-assign]
    assert asyncio.run(cascade.candidates("Корректный текст.")) == []
    assert cascade.metrics().failures == 1
