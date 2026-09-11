"""Маршрутизация стадий локального корректора (v9).

Что было не так в v8
====================
В логах прод-сервера на **каждом** запросе видно::

    Grammar gate hit: skipping expensive GEC specialist for this text
    stages={'rules': 1, 'sage': 1, 'gec': 0, 'draft_tlite': 0, ...}

`LOCAL_GRAMMAR_GATE_CATEGORIES` по умолчанию включал `rule-agreement`,
а правило согласования v8 срабатывало почти на любом тексте (ложно).
Поэтому gate выключал и GEC-специалиста, и обе rescue-модели: служба
деградировала до «сломанное правило + SAGE». Отсюда одновременно оба
симптома из отчёта — «не находит ошибок» и «портит корректный текст».

Архитектура v9
==============
Gate убран. Вместо него — порядок «дешёвое и доказуемое сначала,
генеративное затем, арбитраж в конце»::

    предложения ──┬─ LocalRuleEngine        (детерминированно, µs)
                  ├─ DictionarySpellChecker (детерминированно, мс)
                  ├─ LanguageTool           (rule-based, опционально)
                  ├─ SAGE                   (орфография/пунктуация)
                  └─ Qwen3.5-GEC            (грамматика)
                             │
                  rescue (T-lite / GigaChat) — только если выше ничего
                             │                 не подтверждено
                  CandidateArbiter → голосование и слияние
                             │
                  GenerativeGuard + DecisionEngine → точные правки

Ключевые отличия от v8:

* модели вызываются **по предложениям** с абсолютными смещениями;
* ответ модели нормализуется (`llm_text.sanitize`) до diff-а;
* GEC-специалист получает few-shot из банка эталонов (в v8 retrieval
  был подключён только к rescue-моделям);
* один переиспользуемый `httpx.AsyncClient` вместо клиента на вызов;
* rescue включается по количеству **подтверждённых** кандидатов, а не
  по количеству сырых.
"""

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
from languagetool_stage import LanguageToolStage
from llm_text import sanitize
from local_rules import LocalRuleEngine
from ollama_gec import OllamaGecSpecialist
from russian_quality_models import SageRussianCorrector
from safe_diff import diff_candidates as bounded_diff
from segmentation import Segment, split_sentences, strip_enumeration
from spellcheck import DictionarySpellChecker
from verification import CandidateArbiter, is_deterministic

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
    "A": StackInfo("A", "production: deterministic-first + SAGE + Qwen3.5 GEC + adaptive T-lite rescue", TLITE, False),
    "B": StackInfo("B", "production-candidate: deterministic-first + SAGE + Qwen3.5 GEC + adaptive GigaChat rescue", GIGACHAT, False),
    "X": StackInfo("X", "experimental-fast: deterministic-first + SAGE + Qwen3.5 GEC", "qwen3.5-gec+SAGE", True),
    "Y": StackInfo("Y", "experimental-max: deterministic-first + SAGE + Qwen3.5 GEC + both Ollama rescues", "qwen3.5-gec+SAGE+Ollama", True),
}

STAGE_KEYS = ("rules", "spell", "languagetool", "sage", "gec", "draft_tlite", "draft_giga")


class RetrievalExamples:
    """Банк эталонных пар «неверно → верно» для few-shot подсказок."""

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

    def pairs(self, text: str, top_k: int = 5) -> list[tuple[str, str, str]]:
        if not self.bank or not self.available:
            return []
        try:
            found = [pair for _, pair in self.bank.search_hybrid(text, top_k=top_k)]
        except Exception as exc:
            logger.warning("GEC retrieval search failed: %s", exc)
            return []
        out: list[tuple[str, str, str]] = []
        for pair in found:
            wrong = getattr(pair, "wrong", "")
            right = getattr(pair, "right", "")
            if wrong and right and wrong != right:
                out.append((wrong, right, getattr(pair, "rule", "")))
        return out

    def prompt(self, text: str, top_k: int = 5) -> str:
        pairs = self.pairs(text, top_k)
        if not pairs:
            return ""
        lines = ["ПОХОЖИЕ ЭТАЛОНЫ:"]
        for i, (wrong, right, rule) in enumerate(pairs, 1):
            suffix = f" | {rule}" if rule else ""
            lines.append(f"{i}. {wrong} → {right}{suffix}")
        return "\n".join(lines)


class OllamaDraftClient:
    """Rescue-генератор: полный перепис фрагмента с последующим diff-ом."""

    SYSTEM = (
        "Ты — специализированный корректор русского официально-делового текста. "
        "Ищи только объективные ошибки орфографии, пунктуации, грамматики, "
        "согласования, управления и синтаксиса. Особенно проверяй падеж, число "
        "и управление числительных, местоимений и существительных. "
        "Сохраняй все слова, порядок слов, смысл, термины, числа, переносы строк и структуру. "
        "Не перефразируй и не форматируй текст. Ничего не добавляй и не удаляй, "
        "кроме символов и слов, необходимых для исправления объективной ошибки. "
        "Верни только исправленный текст целиком, без пояснений и без кавычек."
    )

    def __init__(self, model: str, retriever: RetrievalExamples | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = float(os.getenv("HYBRID_LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "30")))
        self.ctx = int(os.getenv("HYBRID_NUM_CTX", "3072"))
        self.predict = int(os.getenv("HYBRID_NUM_PREDICT", "384"))
        self.threads = int(os.getenv("NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.retriever = retriever
        self._client = client

    async def draft(self, text: str, context: str, temperature: float = 0.0) -> str:
        retrieval = self.retriever.prompt(text) if self.retriever else ""
        user = f"Контекст:\n{context[-2200:]}\n\n{retrieval}\n\nТекст:\n{text}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user},
            ],
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
        if self._client is not None:
            response = await self._client.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:  # pragma: no cover - путь без общего клиента
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
        self.protected_words = protected_words or set()
        self.rules = LocalRuleEngine()
        self.speller = DictionarySpellChecker(protected_words=self.protected_words)
        self.retriever = RetrievalExamples()
        self.sage = SageRussianCorrector()
        limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
        self._client: httpx.AsyncClient = httpx.AsyncClient(limits=limits)
        self.gec = OllamaGecSpecialist(self.retriever, self._client)
        self.languagetool = LanguageToolStage()
        self.arbiter = CandidateArbiter()
        self.max_sentences = int(os.getenv("LOCAL_MAX_SENTENCES", "12"))
        self.rescue_mode = os.getenv("LOCAL_RESCUE_MODE", "auto").strip().lower()
        self.min_verified = int(os.getenv("LOCAL_RESCUE_MIN_CANDIDATES", "1"))
        self._calls = 0
        self._stage_calls = {key: 0 for key in STAGE_KEYS}
        self._stage_ms = {key: 0 for key in STAGE_KEYS}
        self._degraded: list[str] = []

    # ------------------------------------------------------------------
    # Инфраструктура
    # ------------------------------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:
        """Общий keep-alive клиент к Ollama.

        v8 создавал новый `AsyncClient` на каждый вызов каждой стадии:
        при 4 стадиях × N предложений это N×4 новых TCP-соединения на
        запрос.
        """
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _timed(self, key: str, coro):
        started = time.perf_counter()
        try:
            return await coro
        finally:
            self._stage_ms[key] += int((time.perf_counter() - started) * 1000)

    def _rescue_models(self) -> list[tuple[str, str]]:
        if self.preset == "X":
            return []
        if self.preset == "Y":
            return [("draft_tlite", TLITE), ("draft_giga", GIGACHAT)]
        if self.preset == "A":
            return [("draft_tlite", TLITE)]
        return [("draft_giga", GIGACHAT)]

    def _segments(self, text: str) -> list[Segment]:
        segments = [strip_enumeration(s) for s in split_sentences(text)]
        segments = [s for s in segments if s.text.strip()]
        return segments[: self.max_sentences]

    # ------------------------------------------------------------------
    # Стадии
    # ------------------------------------------------------------------
    def _deterministic(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        self._stage_calls["rules"] += 1
        started = time.perf_counter()
        out.extend(self.rules.candidates(text))
        self._stage_ms["rules"] += int((time.perf_counter() - started) * 1000)

        if self.speller.available:
            self._stage_calls["spell"] += 1
            started = time.perf_counter()
            out.extend(self.speller.candidates(text))
            self._stage_ms["spell"] += int((time.perf_counter() - started) * 1000)
        return out

    async def _model_candidates(self, text: str, segments: list[Segment]) -> list[EditCandidate]:
        """SAGE и GEC-специалист по предложениям, параллельно."""
        jobs: list[tuple[str, Segment, Any]] = []

        if self.sage.available:
            self._stage_calls["sage"] += 1
            jobs.append((
                "sage", Segment(text, 0),
                self._timed("sage", self.sage.correct_batch([s.text for s in segments])),
            ))
        if self.gec.available:
            for segment in segments:
                self._stage_calls["gec"] += 1
                jobs.append(("gec", segment, self._timed("gec", self.gec.correct(segment.text))))
        if self.languagetool.available:
            self._stage_calls["languagetool"] += 1
            jobs.append((
                "languagetool", Segment(text, 0),
                self._timed("languagetool", self.languagetool.candidates(text)),
            ))

        if not jobs:
            return []
        results = await asyncio.gather(*(job[2] for job in jobs), return_exceptions=True)

        out: list[EditCandidate] = []
        for (stage, segment, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("%s stage failed: %s", stage, result)
                continue
            if stage == "languagetool":
                out.extend(result)
                continue
            if stage == "sage":
                for source, produced in zip(segments, result):
                    cleaned = sanitize(source.text, produced)
                    if cleaned:
                        out.extend(bounded_diff(
                            source.text, cleaned, "sage-spell-punc", 0.90, offset=source.start,
                        ))
                continue
            cleaned = sanitize(segment.text, str(result or ""))
            if cleaned:
                out.extend(bounded_diff(
                    segment.text, cleaned, "russian-gec", 0.96, offset=segment.start,
                ))
        return out

    async def _rescue_candidates(self, text: str, context: str,
                                 segments: list[Segment]) -> list[EditCandidate]:
        models = self._rescue_models()
        if not models:
            return []
        jobs: list[tuple[str, Segment, Any]] = []
        for stage, model in models:
            client = OllamaDraftClient(model, self.retriever, self.client)
            for segment in segments:
                self._stage_calls[stage] += 1
                jobs.append((
                    stage, segment,
                    self._timed(stage, client.draft(segment.text, context, 0.0)),
                ))
        results = await asyncio.gather(*(job[2] for job in jobs), return_exceptions=True)
        out: list[EditCandidate] = []
        for (stage, segment, _), result in zip(jobs, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("%s rescue failed: %s", stage, result)
                continue
            cleaned = sanitize(segment.text, str(result or ""))
            if cleaned:
                out.extend(bounded_diff(
                    segment.text, cleaned, stage, 0.70, offset=segment.start,
                ))
        return out

    # ------------------------------------------------------------------
    def _needs_rescue(self, candidates: list[EditCandidate]) -> bool:
        """Нужен ли второй генеративный проход.

        Критерий — количество **подтверждённых** кандидатов
        (детерминированных либо набравших минимум два независимых
        голоса), а не сырых. Поэтому одиночная правка от GEC-специалиста
        тоже запускает rescue: вторая модель либо подтвердит её (тогда
        уверенность вырастет), либо нет (тогда правка останется
        одиночной и пройдёт только если источнику хватает базового веса).
        В v8 критерий считался по сырым кандидатам, и одно ложное
        правило навсегда закрывало rescue.
        """
        if self.rescue_mode == "never":
            return False
        if self.rescue_mode == "always":
            return True
        verified = [c for c in candidates if is_deterministic(c.category) or c.votes > 1]
        return len(verified) < self.min_verified

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        self._calls += 1
        segments = self._segments(text)
        collected = self._deterministic(text)
        collected += await self._model_candidates(text, segments)
        merged = self.arbiter.merge(collected)
        if self._needs_rescue(merged):
            rescue = await self._rescue_candidates(text, context, segments)
            if rescue:
                merged = self.arbiter.merge(collected + rescue)
        return merged

    async def warmup(self) -> None:
        tasks: list[tuple[str, Any]] = []
        if self.sage.available:
            tasks.append(("sage", self.sage.warmup()))
        if self.gec.available:
            tasks.append(("gec", self.gec.warmup()))
        if self.languagetool.available:
            tasks.append(("languagetool", self.languagetool.warmup()))
        if not tasks:
            return
        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)
        for (stage, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                message = f"{stage}: {type(result).__name__}: {result}"
                self._degraded.append(message)
                logger.warning("Warmup degraded: %s", message)

    def ollama_required(self) -> bool:
        return self.gec.available or bool(self._rescue_models())

    @property
    def degraded(self) -> list[str]:
        return list(dict.fromkeys(self._degraded))

    def metrics(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "calls": self._calls,
            "rule_engine": self.rules.available,
            "spellcheck": self.speller.available,
            "languagetool": self.languagetool.available,
            "retrieval": self.retriever.available,
            "retrieval_count": self.retriever.count,
            "sage": self.sage.metrics().__dict__,
            "russian_gec": self.gec.metrics().__dict__,
            "stage_calls": dict(self._stage_calls),
            "stage_ms": dict(self._stage_ms),
            "rescue_mode": self.rescue_mode,
            "ollama_required": self.ollama_required(),
            "degraded": self.degraded,
        }
