from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from decision_engine import EditCandidate
from local_rules import LocalRuleEngine
from ollama_gec import OllamaGecSpecialist
from russian_quality_models import SageRussianCorrector
from safe_diff import diff_candidates as bounded_diff

logger = logging.getLogger("ai_suggester.hybrid")

TLITE = "t-tech/T-lite-it-2.1:q4_K_M"
GIGACHAT = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS = {
    "A": StackInfo("A", "production: deterministic rules + SAGE + Qwen3.5 GEC + adaptive T-lite rescue", TLITE, False),
    "B": StackInfo("B", "production-candidate: deterministic rules + SAGE + Qwen3.5 GEC + adaptive GigaChat rescue", GIGACHAT, False),
    "X": StackInfo("X", "experimental-fast: deterministic rules + SAGE + Qwen3.5 GEC", "qwen3.5-gec+SAGE", True),
    "Y": StackInfo("Y", "experimental-max: deterministic rules + SAGE + Qwen3.5 GEC + both Ollama rescues", "qwen3.5-gec+SAGE+Ollama", True),
}


class RetrievalExamples:
    def __init__(self) -> None:
        self.bank = None
        self.available = False
        self.count = 0
        try:
            from shared.gec_bank import GecBank
            from shared.rag_store import HashingEmbedder
            root = Path(__file__).resolve().parents[1] / "shared" / "gec_seed"
            configured = os.getenv("GEC_BANK_FILES", "").strip()
            paths = [Path(p.strip()) for p in configured.split(",") if p.strip()] if configured else [
                root / "gec_bank_extended.jsonl", root / "lexify_admin.jsonl"
            ]
            existing = [p for p in paths if p.exists()]
            if not existing:
                return
            self.bank = GecBank(HashingEmbedder(512), bm25_tokenizer="both")
            self.bank.load_jsonl(*existing)
            self.bank.build_index(Path(os.getenv("GEC_BANK_CACHE", "data/gec_bank_hashing.pkl")))
            self.count = len(self.bank)
            self.available = self.count > 0
            logger.info("GEC retrieval: %d pairs ready", self.count)
        except Exception as exc:
            logger.warning("GEC retrieval unavailable: %s", exc)

    def prompt(self, text: str, top_k: int = 5) -> str:
        if not self.bank or not self.available:
            return ""
        try:
            pairs = [pair for _, pair in self.bank.search_hybrid(text, top_k=top_k)]
            lines = ["ПОХОЖИЕ ЭТАЛОНЫ:"]
            for i, pair in enumerate(pairs, 1):
                wrong = getattr(pair, "wrong", "")
                right = getattr(pair, "right", "")
                rule = getattr(pair, "rule", "")
                if wrong and right:
                    suffix = f" | {rule}" if rule else ""
                    lines.append(f"{i}. {wrong} → {right}{suffix}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as exc:
            logger.warning("GEC retrieval search failed: %s", exc)
            return ""


class OllamaDraftClient:
    def __init__(self, model: str, retriever: RetrievalExamples | None = None) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = float(os.getenv("HYBRID_LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "30")))
        self.ctx = int(os.getenv("HYBRID_NUM_CTX", "3072"))
        self.predict = int(os.getenv("HYBRID_NUM_PREDICT", "384"))
        self.threads = int(os.getenv("NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.retriever = retriever

    async def draft(self, text: str, context: str, temperature: float = 0.0) -> str:
        retrieval = self.retriever.prompt(text) if self.retriever else ""
        system = (
            "Ты — строгий корректор русского официально-делового текста. Исправляй только "
            "объективные ошибки орфографии, пунктуации, грамматики, согласования, управления "
            "и синтаксиса. Сохраняй порядок слов, смысл, термины, числа и структуру. "
            "Не переписывай, не переставляй части предложения и не меняй стиль. Верни только текст."
        )
        user = f"Контекст:\n{context[-2200:]}\n\n{retrieval}\n\nТекст:\n{text}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.ctx,
                "num_predict": self.predict,
                "num_thread": self.threads,
                "repeat_penalty": 1.05,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()


class HybridRouter:
    def __init__(self, preset: str, protected_words: set[str] | None = None) -> None:
        self.preset = preset.upper()
        if self.preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={self.preset!r}; expected A, B, X or Y")
        self.info = STACKS[self.preset]
        self.rules = LocalRuleEngine()
        self.retriever = RetrievalExamples()
        self.sage = SageRussianCorrector()
        self.gec = OllamaGecSpecialist()
        self.protected_words = protected_words or set()
        self._calls = 0
        self._stage_calls = {"rules": 0, "sage": 0, "gec": 0, "draft_tlite": 0, "draft_giga": 0}
        self._stage_ms = {key: 0 for key in self._stage_calls}
        self._degraded: list[str] = []

    def _rescue_models(self) -> list[tuple[str, str]]:
        if self.preset == "X":
            return []
        if self.preset == "Y":
            return [("draft_tlite", TLITE), ("draft_giga", GIGACHAT)]
        if self.preset == "A":
            return [("draft_tlite", TLITE)]
        return [("draft_giga", GIGACHAT)]

    @staticmethod
    def _candidate_weight(candidate: EditCandidate) -> float:
        if candidate.category.startswith("rule-"):
            return 1.00
        if candidate.category == "russian-gec":
            return 0.96
        if candidate.category == "sage-spell-punc":
            return 0.92
        if candidate.category.startswith("draft-"):
            return 0.70
        return candidate.confidence

    @staticmethod
    def _needs_rescue(candidates: list[EditCandidate]) -> bool:
        if not candidates:
            return True
        strong = [c for c in candidates if c.category in {"russian-gec", "sage-spell-punc"} or c.category.startswith("rule-")]
        # A large number of low-value punctuation/spelling edits must not suppress
        # a second grammar-oriented pass.
        grammar = [c for c in strong if c.category == "russian-gec" or c.category.startswith("rule-")]
        return len(grammar) < int(os.getenv("LOCAL_RESCUE_MIN_GRAMMAR_CANDIDATES", "1"))

    def _rank_candidates(self, candidates: list[EditCandidate]) -> list[EditCandidate]:
        unique: dict[tuple[str, str], EditCandidate] = {}
        for candidate in candidates:
            key = (candidate.before, candidate.after)
            prev = unique.get(key)
            if prev is None or self._candidate_weight(candidate) > self._candidate_weight(prev):
                unique[key] = candidate
        ranked = list(unique.values())
        ranked.sort(key=lambda c: (self._candidate_weight(c), len(c.before)), reverse=True)
        return ranked

    async def _timed(self, key: str, coro):
        started = time.perf_counter()
        try:
            return await coro
        finally:
            self._stage_ms[key] += int((time.perf_counter() - started) * 1000)

    async def _local_candidates(self, text: str) -> list[EditCandidate]:
        deterministic = self.rules.candidates(text)
        self._stage_calls["rules"] += 1
        jobs: list[tuple[str, Any]] = []
        if self.sage.available:
            self._stage_calls["sage"] += 1
            jobs.append(("sage", self._timed("sage", self.sage.correct(text))))
        if self.gec.available:
            self._stage_calls["gec"] += 1
            jobs.append(("gec", self._timed("gec", self.gec.correct(text))))
        results = await asyncio.gather(*(job[1] for job in jobs), return_exceptions=True)
        candidates: list[EditCandidate] = list(deterministic)
        for (stage, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("%s candidate generator failed: %s", stage, result)
                continue
            draft = str(result or "").strip()
            if stage == "sage":
                candidates.extend(bounded_diff(text, draft, "sage-spell-punc", 0.90))
            elif stage == "gec":
                candidates.extend(bounded_diff(text, draft, "russian-gec", 0.96))
        return self._rank_candidates(candidates)

    async def _rescue_candidates(self, text: str, context: str) -> list[EditCandidate]:
        models = self._rescue_models()
        if not models:
            return []
        jobs = []
        for stage, model in models:
            self._stage_calls[stage] += 1
            jobs.append((stage, self._timed(stage, OllamaDraftClient(model, self.retriever).draft(text, context, 0.0))))
        results = await asyncio.gather(*(job[1] for job in jobs), return_exceptions=True)
        candidates: list[EditCandidate] = []
        for (stage, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("%s rescue failed: %s", stage, result)
                continue
            draft = str(result or "").strip()
            if draft:
                candidates.extend(bounded_diff(text, draft, stage, 0.70))
        return candidates

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        self._calls += 1
        local = await self._local_candidates(text)
        # The specialist GEC pass is mandatory for A/B/X/Y. Generic Ollama is a
        # second opinion only when grammar coverage is still weak.
        if self._needs_rescue(local):
            rescue = await self._rescue_candidates(text, context)
            return self._rank_candidates(local + rescue)
        return local

    async def warmup(self) -> None:
        tasks: list[tuple[str, Any]] = []
        if self.sage.available:
            tasks.append(("sage", self.sage.warmup()))
        if self.gec.available:
            tasks.append(("gec", self.gec.warmup()))
        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)
        for (stage, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("Warmup degraded: %s", message)

    def ollama_required(self) -> bool:
        return True

    @property
    def degraded(self) -> list[str]:
        return list(dict.fromkeys(self._degraded))

    def metrics(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "calls": self._calls,
            "rule_engine": self.rules.available,
            "retrieval": self.retriever.available,
            "retrieval_count": self.retriever.count,
            "sage": self.sage.metrics().__dict__,
            "russian_gec": self.gec.metrics().__dict__,
            "stage_calls": dict(self._stage_calls),
            "stage_ms": dict(self._stage_ms),
            "ollama_required": True,
            "degraded": self.degraded,
        }
