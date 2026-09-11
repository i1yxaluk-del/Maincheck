from __future__ import annotations

import asyncio

from reasoning_cascade import ReasoningCascade


def test_marked_final_is_used_without_slow_extractor_call():
    cascade = ReasoningCascade()
    calls: list[str] = []

    async def fake_chat(model, messages, timeout, num_predict):
        calls.append(model)
        return "<think>После 'ночных' нужно множественное число.</think>\nИСПРАВЛЕННЫЙ ТЕКСТ: После ночных нарядов."

    cascade._chat = fake_chat  # type: ignore[method-assign]
    edits = asyncio.run(cascade.candidates("После ночных наряда."))
    assert calls == [cascade.reasoner]
    assert [(e.before, e.after) for e in edits] == [("наряда", "нарядов")]
    assert cascade.metrics().direct_results == 1
    assert cascade.metrics().extractor_calls == 0


def test_extractor_is_fallback_when_reasoner_has_no_final_marker():
    cascade = ReasoningCascade()
    calls: list[str] = []

    async def fake_chat(model, messages, timeout, num_predict):
        calls.append(model)
        if len(calls) == 1:
            return "Нужно изменить форму существительного."
        return "После ночных нарядов."

    cascade._chat = fake_chat  # type: ignore[method-assign]
    edits = asyncio.run(cascade.candidates("После ночных наряда."))
    assert calls == [cascade.reasoner, cascade.extractor]
    assert [(e.before, e.after) for e in edits] == [("наряда", "нарядов")]
    assert cascade.metrics().extractor_calls == 1


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
