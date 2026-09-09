from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from decision_engine import EditCandidate
from local_rules import LocalRuleEngine
from safe_diff import diff_candidates as bounded_diff

logger = logging.getLogger("ai_suggester.hybrid")

TLITE = "t-tech/T-lite-it-2.1:q4_K_M"
GIGACHAT = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:latest"
# Text-only Russian GEC specialist. It avoids the Qwen3.5 multimodal processor
# dependency (PIL/torchvision) that made the previous X/Y presets unable to start.
GEC_SPECIALIST = os.getenv(
    "GEC_SPECIALIST_MODEL",
    "ReginaNasyrova/checkpoint_150_lora_grpo_upd_reward_GECExplanation-4B-sft-stage1-March2026",
)

DIAG_SYSTEM = """Ты — диагностический модуль корректора русского официально-делового текста.

Не переписывай текст целиком. Найди ВСЕ объективные ошибки в данном фрагменте,
с учётом полного предложения и контекста.

Проверяй отдельно:
- согласование определения/причастия с существительным, включая удалённые и
  однородные конструкции;
- управление падежом и формы после предлогов/глаголов;
- конструкции с числительными и датами;
- орфографию и очевидные опечатки;
- пунктуацию по синтаксической структуре, а не только по соседним словам;
- лишние запятые между сказуемым и его дополнением;
- пропущенные запятые между частями сложного предложения.

НЕ меняй стиль, терминологию, имена, даты, номера и допустимые варианты.
Для каждой ошибки верни минимальную замену exact substring исходного текста.
Не объединяй несколько независимых ошибок в одну длинную замену.
При сомнении лучше не предлагай правку.

Верни только JSON:
{"edits":[{"before":"...","after":"...","confidence":0.0,"category":"grammar|spelling|punctuation|government|agreement","reason":"..."}]}
"""

JUDGE_SYSTEM = """Ты — независимый редактор-контролёр русского официально-делового текста.

Тебе дан исходный текст и локальные кандидаты исправлений.
Для каждого кандидата заново проверь полное предложение, синтаксическую связь,
управление, согласование и пунктуационный контекст.
true только для объективной ошибки с однозначной минимальной заменой.
false для стилистики, допустимого варианта, термина, даты, номера или сомнения.
Не создавай новые исправления.

Верни только JSON: {"accept":[true,false,...]}
"""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS = {
    "A": StackInfo("A", "production: deterministic Russian rules + T-lite diagnostic + GigaChat judge", TLITE, False),
    "B": StackInfo("B", "production-candidate: deterministic Russian rules + GigaChat diagnostic + T-lite judge", GIGACHAT, False),
    "X": StackInfo("X", "experimental: Russian GEC specialist + GigaChat judge", GEC_SPECIALIST, True),
    "Y": StackInfo("Y", "experimental: two specialist GEC drafts + T-lite judge + optional LanguageTool", GEC_SPECIALIST, True),
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

    def prompt(self, text: str, top_k: int = 4) -> str:
        if not self.bank or not self.available:
            return ""
        try:
            pairs = [pair for _, pair in self.bank.search_hybrid(text, top_k=top_k)]
            lines = ["ПОХОЖИЕ ЭТАЛОНЫ (только как ориентир):"]
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
        self.predict = int(os.getenv("HYBRID_NUM_PREDICT", "512"))
        self.threads = int(os.getenv("NUM_THREADS", "16"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
        self.retriever = retriever

    async def call(self, messages: list[dict[str, str]], schema: dict[str, Any], *, temperature: float = 0.0, think: bool = False) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        try:
            data = json.loads(content)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def diagnose(self, text: str, context: str, temperature: float = 0.0, think: bool = False) -> list[EditCandidate]:
        retrieval = self.retriever.prompt(text) if self.retriever else ""
        user = (
            f"КОНТЕКСТ ДОКУМЕНТА:\n{context[-3500:]}\n\n"
            f"{retrieval}\n\nИСПРАВЛЯЕМЫЙ ФРАГМЕНТ:\n{text}"
        )
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array", "maxItems": 12,
                    "items": {
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
                    },
                }
            },
            "required": ["edits"],
            "additionalProperties": False,
        }
        data = await self.call(
            [{"role": "system", "content": DIAG_SYSTEM}, {"role": "user", "content": user}],
            schema,
            temperature=temperature,
            think=think,
        )
        raw = data.get("edits", []) if isinstance(data, dict) else []
        return [
            EditCandidate(
                str(item.get("before", "")),
                str(item.get("after", "")),
                float(item.get("confidence", 0.0)),
                str(item.get("category", "model-diagnostic")),
                str(item.get("reason", "")),
            )
            for item in raw
            if isinstance(item, dict) and item.get("before") and item.get("after")
        ]

    async def judge(self, text: str, candidates: list[EditCandidate]) -> list[bool]:
        payload = [
            {"id": i, "before": c.before, "after": c.after, "category": c.category, "reason": c.reason}
            for i, c in enumerate(candidates)
        ]
        schema = {
            "type": "object",
            "properties": {"accept": {"type": "array", "items": {"type": "boolean"}, "maxItems": 16}},
            "required": ["accept"],
            "additionalProperties": False,
        }
        data = await self.call(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": json.dumps({"text": text, "candidates": payload}, ensure_ascii=False)}],
            schema,
        )
        flags = data.get("accept") if isinstance(data, dict) else None
        return flags if isinstance(flags, list) else []


class LanguageToolVerifier:
    def __init__(self) -> None:
        self.url = os.getenv("LANGUAGETOOL_URL", "").strip().rstrip("/")
        self.language = os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def _matches(self, text: str) -> list[dict[str, Any]]:
        if not self.url:
            return []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(f"{self.url}/v2/check", data={"language": self.language, "text": text})
                response.raise_for_status()
                data = response.json()
                return data.get("matches", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("LanguageTool verifier unavailable: %s", exc)
            return []

    async def supports(self, source: str, corrected: str) -> bool:
        if not self.enabled:
            return True
        before = await self._matches(source)
        after = await self._matches(corrected)
        return len(after) <= len(before)


class HybridRouter:
    def __init__(self, preset: str, protected_words: set[str] | None = None) -> None:
        self.preset = preset.upper()
        if self.preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={self.preset!r}; expected A, B, X or Y")
        self.info = STACKS[self.preset]
        self.rules = LocalRuleEngine()
        self.retriever = RetrievalExamples()
        self.lt = LanguageToolVerifier()
        self.protected_words = protected_words or set()
        self._calls = 0

    def _model_pair(self) -> tuple[str, str]:
        if self.preset == "A":
            return TLITE, GIGACHAT
        if self.preset == "B":
            return GIGACHAT, TLITE
        return GEC_SPECIALIST, GIGACHAT if self.preset == "X" else TLITE

    async def _ollama_diagnostics(self, model: str, text: str, context: str) -> list[EditCandidate]:
        client = OllamaJSON(model, self.retriever)
        return await client.diagnose(text, context, temperature=0.0, think=False)

    async def _specialist_drafts(self, text: str, context: str, count: int) -> list[str]:
        from qwen35_backend import Qwen35Backend
        backend = Qwen35Backend()
        return [await backend.correct(text, context, temperature=t) for t in ([0.10, 0.25][:count])]

    async def candidates(self, text: str, context: str = "") -> list[EditCandidate]:
        self._calls += 1
        deterministic = self.rules.candidates(text)
        generator, judge = self._model_pair()

        if self.preset in {"X", "Y"}:
            drafts = await self._specialist_drafts(text, context, 2 if self.preset == "Y" else 1)
            model_candidates: list[EditCandidate] = []
            for draft in drafts:
                model_candidates.extend(bounded_diff(text, draft, "specialist-draft", 0.76 if len(drafts) == 1 else 0.78))
            if len(drafts) == 2 and drafts[0] == drafts[1]:
                model_candidates.extend(bounded_diff(text, drafts[0], "specialist-consensus", 0.88))
        else:
            model_candidates = await self._ollama_diagnostics(generator, text, context)

        # Deterministic candidates are an independent high-confidence source;
        # an LLM judge is forbidden from deleting them just because it is uncertain.
        judged: list[EditCandidate] = []
        if model_candidates:
            try:
                flags = await OllamaJSON(judge, self.retriever).judge(text, model_candidates[:16])
            except Exception as exc:
                logger.warning("Cross-model judge failed: %s", exc)
                flags = []
            if flags:
                judged = [c for i, c in enumerate(model_candidates[:16]) if i < len(flags) and flags[i] is True]
            else:
                # Transport/schema failure is not a semantic rejection.
                judged = model_candidates[:16]

        unique: dict[tuple[str, str], EditCandidate] = {}
        for c in deterministic + judged:
            key = (c.before, c.after)
            existing = unique.get(key)
            if existing is None or c.confidence > existing.confidence:
                unique[key] = c

        merged = list(unique.values())
        if merged and self.lt.enabled:
            # Optional LT is a quality signal after candidate generation, never the source.
            from decision_engine import DecisionEngine
            candidate_text, _ = DecisionEngine(protected_words=self.protected_words).apply(text, merged)
            if not await self.lt.supports(text, candidate_text):
                merged = [c for c in merged if c.category.startswith("rule-")]
        return merged

    async def warmup(self) -> None:
        if self.preset in {"X", "Y"}:
            from qwen35_backend import Qwen35Backend
            await Qwen35Backend().warmup()
        else:
            model, _ = self._model_pair()
            await OllamaJSON(model, self.retriever).diagnose("Проверка запуска.", "Проверка запуска.")

    def metrics(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "calls": self._calls,
            "rule_engine": self.rules.available,
            "retrieval": self.retriever.available,
            "retrieval_count": self.retriever.count,
            "languagetool_verifier": self.lt.enabled,
        }
