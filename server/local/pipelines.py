from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from decision_engine import DecisionEngine, EditCandidate
from shared.gec_bank import GecBank, build_few_shot_messages
from shared.rag_store import HashingEmbedder

logger = logging.getLogger("ai_suggester.pipelines")

A_MODEL = "t-tech/T-lite-it-2.1:q4_K_M"
B_MODEL = os.getenv("B_MODEL", "hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M")
C_MODEL = os.getenv("C_MODEL", "hf.co/ai-sage/GigaChat-3.1-Lightning-10B-A1.8B-Instruct-GGUF:Q4_K_M")
F_MODEL = os.getenv("F_MODEL", "melsmm/Spell-Corrector-RU-4B")
G_MODEL = os.getenv("G_VERIFIER_MODEL", A_MODEL)
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-/][А-Яа-яЁёA-Za-z]+)*")

A_SYSTEM = """Ты — редактор русского официально-делового текста.

Ищи только реальные локальные ошибки и возвращай только минимальные before -> after edits.

РАЗРЕШЕНО:
- явные опечатки и орфография;
- очевидная пунктуация;
- очевидное согласование и управление.

ЗАПРЕЩЕНО:
- улучшать стиль или перефразировать;
- переписывать предложения или абзацы;
- менять термины, аббревиатуры, имена, названия, номера и даты;
- менять е/ё;
- заменять одну допустимую словоформу на другую без обязательного грамматического основания.

Каждый BEFORE обязан быть точной непрерывной подстрокой исходного текста.
AFTER — только минимальная замена. Не возвращай исправленный текст целиком.
При сомнении верни пустой edits. Максимум 8 правок.
"""

G_SYSTEM = """Ты — консервативный валидатор предложенных локальных правок русского официально-делового текста.
Не создавай новые правки и не переписывай текст. Для каждого кандидата true
только если ошибка очевидна и AFTER однозначно исправляет её. Допустимые
варианты, стиль и спорные формы отклоняй. При сомнении — false."""

F_PROMPT = """Исходный текст:
{TEXT}

Исправь только явные ошибки. Не перестраивай предложения, не меняй порядок слов
и не изменяй грамматические формы без однозначного основания. Сохрани абзацы и
все переносы строк."""


@dataclass(frozen=True)
class StackInfo:
    name: str
    description: str
    model: str
    experimental: bool


STACKS = {
    "A": StackInfo("A", "production: edit-first T-lite + retrieval + morphology", A_MODEL, False),
    "B": StackInfo("B", "experimental: YandexGPT-5-Lite structured edit pipeline", B_MODEL, True),
    "C": StackInfo("C", "experimental: GigaChat 3.1 Lightning structured edit pipeline", C_MODEL, True),
    "F": StackInfo(
        "F",
        "experimental, NOT RECOMMENDED as of 2026-09-08: Spell-Corrector-RU-4B "
        "+ bounded surface gate — on prod regression text the backend model "
        "returns partial rewrites (322/139 chars vs 143-147 char input) instead "
        "of minimal edits, so SurfaceGate correctly rejects everything and the "
        "stack contributes 0 candidates while still costing 12-20s. Keep the "
        "code for future experiments with a different backend model, but do "
        "not switch prod to F until the backend is replaced or benchmarked "
        "positively on your corpus.",
        F_MODEL, True,
    ),
    "G": StackInfo("G", "experimental: local edit detector + T-lite verifier", G_MODEL, True),
}


class MorphologyRescue:
    """High-precision local rescue for obvious modifier+noun agreement."""

    def __init__(self) -> None:
        self.morph = None
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover
            logger.warning("MorphologyRescue unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self.morph is not None

    @staticmethod
    def _is_modifier(parse: Any) -> bool:
        tag = str(parse.tag)
        return any(x in tag for x in ("ADJF", "ADJS", "PRTF", "PRTS"))

    @staticmethod
    def _is_participle(parse: Any) -> bool:
        return "PRTF" in str(parse.tag) or "PRTS" in str(parse.tag)

    @staticmethod
    def _is_noun(parse: Any) -> bool:
        return "NOUN" in str(parse.tag)

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.morph:
            return []
        tokens = list(WORD_RE.finditer(text))
        out: list[EditCandidate] = []
        for left, right in zip(tokens, tokens[1:]):
            modifier, noun = left.group(0), right.group(0)
            if "-" in modifier or "/" in modifier or any(c.isdigit() for c in modifier):
                continue
            modifiers = [p for p in self.morph.parse(modifier) if self._is_modifier(p)]
            nouns = [p for p in self.morph.parse(noun) if self._is_noun(p)]
            if not modifiers or not nouns:
                continue
            compatible = any(
                a.tag.number == n.tag.number
                and a.tag.case == n.tag.case
                and (not a.tag.gender or not n.tag.gender or a.tag.gender == n.tag.gender)
                for a in modifiers for n in nouns
            )
            if compatible:
                continue
            if any(self._is_participle(a) for a in modifiers) and any(n.tag.case == "ablt" for n in nouns):
                continue
            source_parse, noun_parse = modifiers[0], nouns[0]
            # Prefer correcting the modifier when the noun already has a
            # valid form. This covers groups such as "должностного лиц":
            # inflecting only the noun would produce "должностного лица"
            # and hide the number error in the source adjective.
            if source_parse.tag.number and noun_parse.tag.number and source_parse.tag.case:
                target = noun_parse.tag.number
                modifier_form = source_parse.inflect({target, noun_parse.tag.case})
                if modifier_form and modifier_form.word != modifier:
                    after = modifier_form.word
                    if modifier[:1].isupper():
                        after = after[:1].upper() + after[1:]
                    out.append(EditCandidate(modifier, after, 0.97, "agreement", "согласование определения с существительным"))
                    continue
            grammemes = {g for g in (source_parse.tag.number, source_parse.tag.case) if g}
            if not grammemes:
                continue
            inflected = noun_parse.inflect(grammemes)
            if not inflected or inflected.word == noun:
                continue
            after = inflected.word
            if noun[:1].isupper():
                after = after[:1].upper() + after[1:]
            out.append(EditCandidate(noun, after, 0.96, "agreement", "очевидное согласование с определением"))
        return out


class RetrievalExamples:
    """Fast offline retrieval: hashing cosine + BM25 word/char-trigram fusion."""

    def __init__(self) -> None:
        self.bank = None
        self.count = 0
        self.available = False
        try:
            root = Path(__file__).resolve().parents[1] / "shared" / "gec_seed"
            configured = os.getenv("GEC_BANK_FILES", "").strip()
            paths = [Path(p.strip()) for p in configured.split(",") if p.strip()] if configured else [
                root / "gec_bank_extended.jsonl",
                root / "lexify_admin.jsonl",
            ]
            existing = [p for p in paths if p.exists()]
            if not existing:
                return
            self.bank = GecBank(HashingEmbedder(512), bm25_tokenizer="both")
            self.bank.load_jsonl(*existing)
            self.bank.build_index(Path(os.getenv("GEC_BANK_CACHE", "data/gec_bank_hashing.pkl")))
            self.count = len(self.bank)
            self.available = self.count > 0
            logger.info("RetrievalExamples: %d pairs ready", self.count)
        except Exception as exc:  # pragma: no cover
            logger.warning("RetrievalExamples unavailable: %s", exc)

    def examples(self, text: str, top_k: int = 2):
        if not self.bank or not self.available:
            return []
        try:
            return [pair for _, pair in self.bank.search_hybrid(text, top_k=top_k)]
        except Exception as exc:  # pragma: no cover
            logger.warning("Retrieval search failed: %s", exc)
            return []


class SurfaceGate:
    """Fail-closed gate for full-text correction experiments."""

    def __init__(self) -> None:
        self.morph = None
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
        except Exception as exc:  # pragma: no cover
            logger.warning("SurfaceGate unavailable: %s", exc)

    def _signature(self, word: str) -> tuple[str, ...] | None:
        if not self.morph:
            return None
        parses = self.morph.parse(word)
        if not parses:
            return None
        tag = parses[0].tag
        attrs = ("POS", "case", "number", "gender", "tense", "person", "voice", "aspect", "mood", "animacy")
        return tuple(f"{name}={getattr(tag, name, None)}" for name in attrs)

    def accept(self, before: str, after: str) -> bool:
        if not before or not after or before == after or "\n" in before or "\n" in after:
            return False
        bw, aw = WORD_RE.findall(before), WORD_RE.findall(after)
        if len(bw) != len(aw):
            return False
        if not bw:
            return True
        if not self.morph:
            return all(a.casefold() == b.casefold() for a, b in zip(bw, aw))
        return all(self._signature(a) == self._signature(b) for a, b in zip(bw, aw))


def surface_candidates(source: str, corrected: str) -> list[EditCandidate]:
    if source == corrected or not source or not corrected:
        return []
    if source.count("\n") != corrected.count("\n"):
        return []
    if len(source.split("\n\n")) != len(corrected.split("\n\n")):
        return []
    out: list[EditCandidate] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, source, corrected, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            before, after = source[i1:i2], corrected[j1:j2]
            if 0 < len(before) <= 80 and 0 < len(after) <= 80 and "\n" not in before and "\n" not in after:
                out.append(EditCandidate(before, after, 0.80, "surface-model", "bounded surface diff"))
            continue
        left, right = max(0, i1 - 12), min(len(source), i2 + 12)
        left_b, right_b = max(0, j1 - 12), min(len(corrected), j2 + 12)
        before, after = source[left:right], corrected[left_b:right_b]
        if len(before) <= 80 and len(after) <= 80 and "\n" not in before and "\n" not in after:
            out.append(EditCandidate(before, after, 0.78, "surface-model", "bounded punctuation context diff"))
    return out


class TliteClient:
    def __init__(self, retriever: RetrievalExamples) -> None:
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("LLM_MODEL", A_MODEL)
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "192"))
        self.num_threads = int(os.getenv("NUM_THREADS", "28"))
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "24h")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
        self.retriever = retriever

    @staticmethod
    def schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"edits": {"type": "array", "maxItems": 8, "items": {
                "type": "object",
                "properties": {
                    "before": {"type": "string"}, "after": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "category": {"type": "string"}, "reason": {"type": "string"},
                },
                "required": ["before", "after", "confidence", "category", "reason"],
                "additionalProperties": False,
            }}},
            "required": ["edits"], "additionalProperties": False,
        }

    @staticmethod
    def verify_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"accept": {"type": "array", "maxItems": 8, "items": {"type": "boolean"}}},
            "required": ["accept"], "additionalProperties": False,
        }

    async def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model, "messages": messages, "stream": False, "format": schema,
            "think": False, "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                         "num_predict": self.num_predict, "num_thread": self.num_threads,
                         "repeat_penalty": 1.03},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "{}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    async def candidates(self, text: str, context: str = "", protected_words: set[str] | None = None) -> list[EditCandidate]:
        user = f"ТЕКСТ ДЛЯ ПРОВЕРКИ:\n{text}"
        if context:
            user = f"КРАТКИЙ КОНТЕКСТ:\n{context[:800]}\n\n{user}"
        if protected_words:
            user += "\n\nЗАЩИЩЁННЫЕ ТЕРМИНЫ:\n" + ", ".join(sorted(protected_words, key=str.casefold))
        examples = self.retriever.examples(text, top_k=2)
        messages = build_few_shot_messages(A_SYSTEM, user, examples) if examples else [
            {"role": "system", "content": A_SYSTEM}, {"role": "user", "content": user}
        ]
        messages[-1]["content"] += "\n\nВерни ТОЛЬКО JSON по схеме."
        return DecisionEngine.parse(await self.chat_json(messages, self.schema()))

    async def verify(self, text: str, candidates: list[EditCandidate]) -> list[EditCandidate]:
        """Просит T-lite подтвердить/отклонить уже найденные детерминированные
        кандидаты (numeral_noun/adj_noun/MorphologyRescue).

        v2.4.1 FIX (2026-09-08): раньше при ЛЮБОЙ проблеме с ответом Ollama —
        сетевая ошибка, невалидный JSON, отсутствие/усечение поля `accept` —
        функция возвращала `[]`, т.е. **отбрасывала абсолютно все** уже
        провалидированные детерминированные кандидаты, включая
        высокоуверенные (confidence=0.90-0.96, чистый pymorphy3/natasha
        сигнал без всякого участия LLM). Это fail-closed поведение, и оно
        воспроизводимо ломало recall: на тесте
        «должностного лиц ... служебно-боевой деятельностей» stack A
        (тот же детерминированный LocalEditTagger, БЕЗ verify-прослойки)
        находил и принимал «лиц→лица», а stack G на ТОМ ЖЕ тексте
        возвращал candidates=0 — притом что pymorphy3 детерминирован и
        обязан был найти тот же кандидат. Единственное объяснение —
        verify() тихо выкинул его из-за сбоя парсинга JSON от T-lite,
        а не из-за содержательного «нет».
        Инфраструктурный сбой связи с LLM — это НЕ вердикт «правка
        неверна», и не должен трактоваться как отказ. Теперь при любом
        сбое коммуникации/парсинга функция **fail-open**: возвращает
        исходные детерминированные кандидаты как есть (как будто
        verify-шага не было) и громко логирует это как WARNING, чтобы
        сбой был виден в мониторинге, а не маскировался под «ошибок нет».
        Единственный путь реально ОТКЛОНИТЬ кандидата — explicit
        `accept[i] == false` от модели после успешного парсинга JSON.
        """
        if not candidates:
            return []
        subset = candidates[:8]
        payload = [
            {"id": i, "before": c.before, "after": c.after, "category": c.category}
            for i, c in enumerate(subset)
        ]
        messages = [
            {"role": "system", "content": G_SYSTEM},
            {"role": "user", "content": json.dumps({"text": text, "candidates": payload}, ensure_ascii=False)},
        ]
        try:
            data = await self.chat_json(messages, self.verify_schema())
        except Exception as exc:  # noqa: BLE001 — сеть/таймаут/HTTP-ошибка Ollama
            logger.warning(
                "Stack G verify(): вызов T-lite провалился (%s: %s) — fail-open, "
                "оставляю %d детерминированных кандидатов без LLM-подтверждения "
                "вместо силового отказа",
                type(exc).__name__, exc, len(subset),
            )
            return subset

        flags = data.get("accept") if isinstance(data, dict) else None
        if not isinstance(flags, list) or not flags:
            logger.warning(
                "Stack G verify(): T-lite вернул нераспознаваемый JSON (%r) — "
                "fail-open, оставляю %d детерминированных кандидатов",
                data, len(subset),
            )
            return subset

        accepted: list[EditCandidate] = []
        for i, c in enumerate(subset):
            if i >= len(flags):
                logger.warning(
                    "Stack G verify(): нет вердикта для кандидата %d («%s»→«%s»), "
                    "оставляю без подтверждения", i, c.before, c.after,
                )
                accepted.append(c)
                continue
            if bool(flags[i]):
                accepted.append(c)
            else:
                logger.info(
                    "Stack G verify(): T-lite явно отклонил «%s»→«%s»",
                    c.before, c.after,
                )
        return accepted


class FBackend:
    def __init__(self) -> None:
        self.model_id = F_MODEL
        self._tokenizer = None
        self._model = None
        self.surface_gate = SurfaceGate()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Experimental[F]: loading model=%s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=os.getenv("EXPERIMENTAL_DEVICE_MAP", "auto"), low_cpu_mem_usage=True)
        self._model.eval()
        logger.info("Experimental[F]: model ready")

    def _device(self):
        assert self._model is not None
        return next(self._model.parameters()).device

    def correct(self, text: str) -> list[EditCandidate]:
        self._load()
        import torch
        assert self._tokenizer is not None and self._model is not None
        encoded = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": F_PROMPT.format(TEXT=text)}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        device = self._device()
        inputs = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
        input_ids = inputs["input_ids"]
        # v2.4.1: было min(192, max(64, len(text)//2+32)) — на реальном
        # прод-тексте (147 chars) это давало потолок ~105 токенов, и модель
        # обрезалась/уходила в пересказ вместо точечной правки (наблюдалось
        # 322 и 139 сгенерированных символов на 147/143-символьный вход —
        # то распухание, то обрыв). Подняли пол и потолок; это НЕ чинит
        # основную проблему (модель генерирует не точечные правки, а
        # пересказ — SurfaceGate тогда справедливо режет всё в 0), но
        # снижает шанс, что сама причина — банальный обрыв генерации.
        max_new = min(320, max(96, len(text) // 2 + 48))
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.1, top_p=0.7, repetition_penalty=1.02)
        generated = self._tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        raw = surface_candidates(text, generated)
        safe = [c for c in raw if self.surface_gate.accept(c.before, c.after)]
        logger.info(
            "Experimental[F]: input_chars=%d generated_chars=%d raw=%d safe=%d "
            "(raw=0 обычно значит модель переписала/сократила текст вместо "
            "точечной правки — SurfaceGate отклоняет диффы длиннее 80 символов "
            "на сегмент; см. Инструкции/ЖУРНАЛ, стек F помечен как не "
            "рекомендованный к прод-использованию до замены backend-модели)",
            len(text), len(generated), len(raw), len(safe),
        )
        return safe


class LocalEditTagger:
    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.rescue = MorphologyRescue()

    def candidates(self, text: str) -> list[EditCandidate]:
        out: list[EditCandidate] = []
        if self.detector is not None and getattr(self.detector, "available", False):
            try:
                for e in self.detector.detect_errors(text):
                    if e.before and e.suggestion and e.before != e.suggestion:
                        out.append(EditCandidate(e.before, e.suggestion, 0.90, e.kind, e.explanation))
            except Exception as exc:
                logger.warning("MorphDetector failed: %s", exc)
        out.extend(self.rescue.candidates(text))
        unique: dict[tuple[str, str], EditCandidate] = {}
        for c in out:
            unique[(c.before, c.after)] = c
        return list(unique.values())


class LanguageToolTagger:
    """Converts a local LanguageTool server response into safe candidates."""

    def __init__(self) -> None:
        self.client = None
        if os.getenv("LANGUAGETOOL_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            from shared.languagetool_client import LanguageToolClient, _parse_csv_env

            self.client = LanguageToolClient(
                url=os.getenv("LANGUAGETOOL_URL", "http://localhost:8081"),
                language=os.getenv("LANGUAGETOOL_LANGUAGE", "ru-RU"),
                enabled_categories=_parse_csv_env(os.getenv("LANGUAGETOOL_ENABLED_CATEGORIES")),
                disabled_categories=_parse_csv_env(os.getenv("LANGUAGETOOL_DISABLED_CATEGORIES")),
                disabled_rules=_parse_csv_env(os.getenv("LANGUAGETOOL_DISABLED_RULES")),
                timeout=float(os.getenv("LANGUAGETOOL_TIMEOUT", "10")),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LanguageTool unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return bool(self.client and self.client.available)

    def candidates(self, text: str) -> list[EditCandidate]:
        if not self.client:
            return []
        try:
            return [
                EditCandidate(m.before, m.suggestion, 0.88, f"languagetool:{m.category_id}", m.message)
                for m in self.client.check(text)
                if m.before and m.suggestion and m.before != m.suggestion
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning("LanguageTool candidate generation failed: %s", exc)
            return []


class StackRouter:
    def __init__(self, preset: str, morph_detector: Any) -> None:
        if preset not in STACKS:
            raise RuntimeError(f"Unsupported LLM_PRESET={preset!r}; expected A, B, C, F or G")
        self.preset = preset
        self.info = STACKS[preset]
        self.retriever = RetrievalExamples()
        self.tlite = TliteClient(self.retriever)
        self.tlite.model = self.info.model
        self.tagger = LocalEditTagger(morph_detector)
        self.language_tool = LanguageToolTagger()
        self.f_backend = FBackend() if preset == "F" else None

    async def warmup(self) -> None:
        if self.preset == "F":
            await asyncio.to_thread(self.f_backend._load)  # type: ignore[union-attr]
            return
        await self.tlite.chat_json(
            [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": "Контрольный текст без ошибок. Верни пустой edits."}],
            self.tlite.schema(),
        )

    async def candidates(self, text: str, context: str = "", protected_words: set[str] | None = None) -> list[EditCandidate]:
        if self.preset in {"A", "B", "C"}:
            local = self.tagger.candidates(text)
            lt = await asyncio.to_thread(self.language_tool.candidates, text)
            generated = [
                EditCandidate(c.before, c.after, c.confidence, f"model:{c.category}", c.reason)
                for c in await self.tlite.candidates(text, context, protected_words)
            ]
            return local + lt + generated
        if self.preset == "G":
            local = self.tagger.candidates(text)
            local.extend(await asyncio.to_thread(self.language_tool.candidates, text))
            return await self.tlite.verify(text, local)
        return await asyncio.to_thread(self.f_backend.correct, text)  # type: ignore[union-attr]

    def metrics(self) -> dict[str, Any]:
        return {
            "retrieval_enabled": self.retriever.available,
            "retrieval_pairs": self.retriever.count,
            "morphology_rescue_enabled": self.tagger.rescue.available,
            "languagetool_enabled": self.language_tool.client is not None,
            "languagetool_available": self.language_tool.available,
        }
