from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from decision_engine import EditCandidate
from local_rules import LocalRuleEngine
from russian_quality_models import SageRussianCorrector
from safe_diff import diff_candidates as bounded_diff

logger = logging.getLogger("ai_suggester.hybrid")

TLITE = "t-tech/T-lite-it-2.1:q4_K_M"
GIGACHAT = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"

DIAG_SYSTEM = """Ты — модуль обнаружения ошибок русского официально-делового текста.

Твоя задача не переписывать текст, а максимально полно найти реальные ошибки.
Проверяй весь контекст предложения и различай:
- орфографию и опечатки;
- пунктуацию и границы частей предложения;
- согласование слов и числительных;
- управление и падежные формы;
- формы глаголов и причастий;
- синтаксические ошибки;
- очевидные повторы, пропуски слов и явные нарушения нормы.

Для каждой ошибки верни минимальную точную замену исходной строки. Не исправляй
стиль без ошибки, термины, номера, названия и допустимые варианты. Не объединяй
независимые ошибки. При сомнении лучше дать меньше кандидатов.

Верни только JSON:
{"edits":[{"before":"...","after":"...","confidence":0.0,"category":"grammar|spelling|punctuation|government|agreement|style","reason":"..."}]}
"""

DRAFT_SYSTEM = """Ты — специализированный редактор русского официально-делового текста.

Исправь ВСЕ объективные орфографические, пунктуационные, грамматические,
синтаксические и очевидные лексические ошибки в исходном тексте.
Сохрани смысл, термины, имена собственные, числа и факты. Не переписывай
предложения ради красоты и не заменяй допустимые варианты. Не добавляй и не
удаляй информацию. Верни только исправленный текст, без комментариев.
"""

JUDGE_SYSTEM = """Ты — строгий контролёр русского текста.

Проверь кандидаты исправлений относительно исходного предложения. Прими только
объективные исправления орфографии, пунктуации, грамматики или синтаксиса.
Отклоняй стилистическую вкусовщину, допустимые варианты и изменения смысла.
Не создавай новых исправлений.
Верни только JSON: {"accept":[true,false,...]}
"""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS = {
    "A": StackInfo("A", "production: SAGE spell/punctuation + T-lite draft/diagnostic + GigaChat judge", TLITE, False),
    "B": StackInfo("B", "production-candidate: SAGE spell/punctuation + GigaChat draft/diagnostic + T-lite judge", GIGACHAT, False),
    "X": StackInfo("X", "experimental: SAGE + GigaChat multi-candidate cascade", GIGACHAT, True),
    "Y": StackInfo("Y", "experimental: SAGE + two T-lite correction candidates + GigaChat judge", TLITE, True),
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
            logger.info("Hybrid retrieval: %d pairs ready", self.count)
        except Exception as exc:
            logger.warning("Hybrid retrieval unavailable: %s", exc)

    def prompt(self, text: str, top_k: int = 6) -> str:
        if not self.bank or not self.available:
            return ""
        try:
            pairs = [pair for _, pair in self.bank.search_hybrid(text, top_k=top_k)]
            lines = ["ПОХОЖИЕ ЭТАЛОНЫ (только как подсказка):"]
            for i, pair in enumerate(pairs, 1):
                wrong = getattr(pair, "wrong", "")
                right = getattr(pair, "right", "")
                rule = getattr(pair, "rule", "")
                if wrong and right:
                    suffix = f" | {rule}" if rule else ""
                    lines.append(f"{i}. {wrong} → {right}{suffix}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as exc:
            logger.warning("Hybrid retrieval search failed: %s", exc)
            return ""


class OllamaJSON:
    def __init__(self, model: str, retriever: RetrievalExamples | None = None) -> None:
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = float(os.getenv("HYBRID_LLM_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "300")))
        self.ctx = int(os.getenv("HYBRID_NUM_CTX", "4096"))
        self.predict = int(os.getenv("HYBRID_NUM_PREDICT", "768"))
        self.threads = int(os.getenv("NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.retriever = retriever

    async def call(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None, *, temperature: float = 0.0, think: bool = False) -> dict[str, Any] | str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self.ctx,
                "num_predict": self.predict,
                "num_thread": self.threads,
                "repeat_penalty": 1.02,
            },
        }
        if schema is not None:
            payload["format"] = schema
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
        if schema is None:
            return str(content)
        try:
            data = json.loads(content)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _user_prompt(self, text: str, context: str) -> str:
        retrieval = self.retriever.prompt(text) if self.retriever else ""
        return f"КОНТЕКСТ ДОКУМЕНТА:\n{context[-4500:]}\n\n{retrieval}\n\nИСХОДНЫЙ ФРАГМЕНТ:\n{text}"

    async def diagnose(self, text: str, context: str, temperature: float = 0.0) -> list[EditCandidate]:
        schema = {
            "type": "object",
            "properties": {
                "edits": {"type": "array", "maxItems": 16, "items": {
                    "type": "object",
                    "properties": {
                        "before": {"type": "string"},
                        "after": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "category": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["before", "after", "confidence", "category", "reason"],
                    "additionalProperties": False,
                }},
            },
            "required": ["edits"],
            "additionalProperties": False,
        }
        data = await self.call(
            [{"role": "system", "content": DIAG_SYSTEM}, {"role": "user", "content": self._user_prompt(text, context)}],
            schema,
            temperature=temperature,
        )
        raw = data.get("edits", []) if isinstance(data, dict) else []
        out: list[EditCandidate] = []
        for item in raw:
            if isinstance(item, dict) and item.get("before") and item.get("after"):
                try:
                    out.append(EditCandidate(
                        str(item["before"]), str(item["after"]), float(item.get("confidence", 0.0)),
                        str(item.get("category", "model-diagnostic")), str(item.get("reason", "")),
                    ))
                except (TypeError, ValueError):
                    continue
        return out

    async def draft(self, text: str, context: str, temperature: float = 0.0) -> str:
        result = await self.call(
            [{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": self._user_prompt(text, context)}],
            None,
            temperature=temperature,
        )
        return str(result or "").strip()

    async def judge(self, text: str, candidates: list[EditCandidate]) -> list[bool]:
        payload = [
            {"id": i, "before": c.before, "after": c.after, "category": c.category, "reason": c.reason}
            for i, c in enumerate(candidates)
        ]
        schema = {
            "type": "object",
            "properties": {"accept": {"type": "array", "items": {"type": "boolean"}, "maxItems": 32}},
            "required": ["accept"],
            "additionalProperties": False,
        }
        data = await self.call(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": json.dumps({"text": text, "candidates": payload}, ensure_ascii=False)}],
            schema,
        )
        flags = data.get("accept") if isinstance(data, dict) else None
        return flags if isinstance(flags, list) else []


class HybridRouter:
    def __init__(self, preset: str, protected_words: set[str] | None = None) -> None:
        self.preset = preset.upper()
        if self.preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={self.preset!r}; expected A, B, X or Y")
        self.info = STACKS[self.preset]
        self.rules = LocalRuleEngine()
        self.retriever = RetrievalExamples()
        self.sage = SageRussianCorrector()
        self.protected_words = protected_words or set()
        self._calls = 0
        self._stage_calls = {"sage": 0, "draft": 0, "diagnostic": 0, "judge": 0}

    def _primary_model(self) -> str:
        if self.preset in {"A", "X"}:
            return TLITE if self.preset == "A" else GIGACHAT
        return GIGACHAT if self.preset == "B" else TLITE

    def _judge_model(self) -> str:
        if self.preset in {"A", "X"}:
            return GIGACHAT
        return TLITE

    async def _ollama(self) -> OllamaJSON:
        return OllamaJSON(self._primary_model(), self.retriever)

    async def _generate_model_candidates(self, text: str, context: str) -> list[EditCandidate]:
        model = await self._ollama()
        draft_count = 2 if self.preset == "Y" else 1
        temps = [0.0, 0.18][:draft_count]
        self._stage_calls["draft"] += draft_count
        self._stage_calls["diagnostic"] += 1
        drafts_task = asyncio.gather(*(model.draft(text, context, t) for t in temps), return_exceptions=True)
        diag_task = model.diagnose(text, context, temperature=0.0)
        draft_results, diag_results = await asyncio.gather(drafts_task, diag_task)

        candidates: list[EditCandidate] = list(diag_results)
        for i, draft in enumerate(draft_results):
            if isinstance(draft, Exception) or not draft:
                logger.warning("Draft generation failed: %s", draft)
                continue
            source_tag = "draft-primary" if i == 0 else "draft-secondary"
            candidates.extend(bounded_diff(text, draft, source_tag, 0.74 if i == 0 else 0.77))
        return candidates

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        self._calls += 1
        deterministic = self.rules.candidates(text)

        specialist_candidates: list[EditCandidate] = []
        if self.sage.available:
            try:
                self._stage_calls["sage"] += 1
                sage_draft = await self.sage.correct(text)
                specialist_candidates.extend(bounded_diff(text, sage_draft, "sage-spell-punc", 0.82))
            except Exception as exc:
                logger.warning("SAGE candidate generation failed: %s", exc)

        model_candidates = await self._generate_model_candidates(text, context)

        judge_client = OllamaJSON(self._judge_model(), self.retriever)
        self._stage_calls["judge"] += 1
        to_judge = specialist_candidates + model_candidates
        judged: list[EditCandidate] = []
        if to_judge:
            try:
                flags = await judge_client.judge(text, to_judge[:32])
            except Exception as exc:
                logger.warning("Cross-model judge failed: %s", exc)
                flags = []
            if flags:
                judged = [c for i, c in enumerate(to_judge[:32]) if i < len(flags) and flags[i] is True]
            else:
                judged = to_judge[:32]

        unique: dict[tuple[str, str], EditCandidate] = {}
        # Local rule candidates are kept independently; the model judge cannot veto them.
        for c in deterministic + judged:
            key = (c.before, c.after)
            existing = unique.get(key)
            if existing is None or c.confidence > existing.confidence:
                unique[key] = c
        return list(unique.values())

    async def warmup(self) -> None:
        if self.sage.available:
            await self.sage.warmup()
        model = OllamaJSON(self._primary_model(), self.retriever)
        await model.diagnose("Проверка запуска.", "Проверка запуска.")

    def metrics(self) -> dict[str, Any]:
        sage_stats = self.sage.metrics()
        return {
            "preset": self.preset,
            "calls": self._calls,
            "rule_engine": self.rules.available,
            "retrieval": self.retriever.available,
            "retrieval_count": self.retriever.count,
            "sage_corrector_enabled": sage_stats.enabled,
            "sage_corrector_loaded": sage_stats.loaded,
            "sage_corrector_model": sage_stats.model,
            "sage_corrector_calls": sage_stats.calls,
            "stage_calls": dict(self._stage_calls),
        }
