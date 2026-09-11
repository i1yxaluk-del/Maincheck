from __future__ import annotations

import asyncio

from decision_engine import DecisionEngine
from rupunct_stage import RuPunctStage


class FakeClassifier:
    def __init__(self, labels):
        self.labels = labels
    def __call__(self, text):
        words = text.split()
        assert len(words) == len(self.labels)
        return [
            {"word": word, "entity_group": label, "score": 0.99}
            for word, label in zip(words, self.labels)
        ]


def stage_with(labels):
    stage = RuPunctStage()
    stage._classifier = FakeClassifier(labels)
    return stage


def test_adds_missing_comma_without_regenerating_text():
    text = "Роль изучена в организации и отражена в отчете."
    labels = ["LOWER_O", "LOWER_COMMA", "LOWER_O", "LOWER_O", "LOWER_O", "LOWER_O", "LOWER_O", "LOWER_O"]
    candidates = asyncio.run(stage_with(labels).candidates(text))
    corrected, _ = DecisionEngine(min_confidence=0.5).apply(text, candidates)
    assert corrected == "Роль изучена, в организации и отражена в отчете."


def test_removes_high_confidence_redundant_comma():
    text = "Изучена роль лица, в организации службы."
    labels = ["LOWER_O"] * 6
    candidates = asyncio.run(stage_with(labels).candidates(text))
    corrected, _ = DecisionEngine(min_confidence=0.5).apply(text, candidates)
    assert corrected == "Изучена роль лица в организации службы."


def test_preserves_visual_line_breaks():
    text = "Изучена\nроль лица,\nв организации службы."
    labels = ["LOWER_O"] * 6
    candidates = asyncio.run(stage_with(labels).candidates(text))
    corrected, _ = DecisionEngine(min_confidence=0.5).apply(text, candidates)
    assert corrected == "Изучена\nроль лица\nв организации службы."
