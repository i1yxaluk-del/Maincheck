"""
Sage-95M post-валидатор (v1.8c)
================================

Подключает компактную русскую GEC-модель `ai-forever/sage-fredt5-distilled-95m`
как второе мнение поверх T-lite. Идея:

  1. T-lite уже сгенерировал ===CORRECTED=== + ===CHANGES===.
  2. Sage запускается **один раз** на исходном тексте и тоже производит
     свою скорректированную версию (sage_corrected).
  3. Для каждой пары (before → after) из T-lite-CHANGES смотрим, согласен
     ли sage с этой правкой:
       - **AGREE**:    `after` встречается в sage_corrected → правку оставляем.
       - **DISAGREE**: `before` остался в sage_corrected (sage не стал
                       править эту позицию) → правка возможно ложная.
  4. По домену (`SAGE_VALIDATOR_DOMAIN`):
       - **admin** (default): дропаем правку **только** при DISAGREE.
                              Recall важнее — лучше пропустить FP, чем
                              удалить TP.
       - **general**: дропаем правку при DISAGREE **или** UNKNOWN
                      (sage не оставил ни before, ни after).

Sage НЕ заменяет T-lite — это **узкий пост-фильтр**. Morph-детектор (v1.8a)
работает до и независимо, sage его правки не трогает (мы фильтруем только
правки от модели, не от детектора).

Зависимости:
  - `transformers>=4.40` (опционально, lazy import)
  - `torch>=2.2`         (опционально, lazy import)
  - `huggingface_hub`    (опционально, lazy import)

Если что-то из этого не установлено или модель не загрузилась —
валидатор **молча отключается** (loader.is_available() = False) и
весь пайплайн работает как до v1.8c. Это критично для прода: апгрейд
по схеме «pull + restart» не должен падать, если кто-то забыл установить
зависимости.

ВАЖНО (история):
  Sage уже тестировалась как primary corrector (см. ЖУРНАЛ_v1.6.md, PoC от
  30 апреля 2026): на структурной пунктуации она не работает (0/3 запятых),
  стабильно галлюцинирует числа («2025» → «2015»), обучена на типо/орфо
  ошибках из Wikipedia. Поэтому в v1.8c она используется ТОЛЬКО как
  узкий пост-фильтр для класса «орфография» (см. SAGE_VALIDATOR_CATEGORIES),
  и по умолчанию работает в DRY-RUN режиме (только логирует verdict'ы,
  ничего не дропает). Включить реальный фильтр — через
  SAGE_VALIDATOR_MODE=enforce + предварительный анализ логов.

ENV:
  SAGE_VALIDATOR_ENABLED=false       (default; нужно явно включить)
  SAGE_VALIDATOR_MODE=dryrun         (dryrun | enforce; default dryrun — только логи)
  SAGE_VALIDATOR_DOMAIN=admin        (admin | general; default admin)
  SAGE_VALIDATOR_CATEGORIES=орфограф (comma-separated substring filter
                                       по «категории» CHANGES-строки; пустая
                                       строка = все категории. Default: только
                                       орфография — sage обучена именно на ней)
  SAGE_VALIDATOR_MODEL=ai-forever/sage-fredt5-distilled-95m
  SAGE_VALIDATOR_DEVICE=cpu          (cpu | cuda; default cpu)
  SAGE_VALIDATOR_MAX_INPUT_LEN=512   (токены, sage-fredt5 контекст ~512)
  SAGE_VALIDATOR_WARMUP=true         (один forward pass при старте)
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "ai-forever/sage-fredt5-distilled-95m"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class SageConfig:
    enabled: bool
    mode: str  # "dryrun" | "enforce"
    domain: str  # "admin" | "general"
    categories: tuple[str, ...]  # substring filter по комментарию правки
    model_name: str
    device: str
    max_input_len: int
    warmup: bool

    @classmethod
    def from_env(cls) -> "SageConfig":
        domain = (os.getenv("SAGE_VALIDATOR_DOMAIN") or "admin").strip().lower()
        if domain not in ("admin", "general"):
            logger.warning(
                "SAGE_VALIDATOR_DOMAIN=%r неизвестно, использую 'admin'", domain
            )
            domain = "admin"
        mode = (os.getenv("SAGE_VALIDATOR_MODE") or "dryrun").strip().lower()
        if mode not in ("dryrun", "enforce"):
            logger.warning(
                "SAGE_VALIDATOR_MODE=%r неизвестно, использую 'dryrun'", mode
            )
            mode = "dryrun"
        cats_raw = os.getenv("SAGE_VALIDATOR_CATEGORIES")
        if cats_raw is None:
            # default — только орфография (sage именно на ней обучена)
            cats: tuple[str, ...] = ("орфограф",)
        else:
            cats = tuple(
                c.strip().lower() for c in cats_raw.split(",") if c.strip()
            )
        return cls(
            enabled=_env_bool("SAGE_VALIDATOR_ENABLED", False),
            mode=mode,
            domain=domain,
            categories=cats,
            model_name=(os.getenv("SAGE_VALIDATOR_MODEL") or DEFAULT_MODEL).strip(),
            device=(os.getenv("SAGE_VALIDATOR_DEVICE") or "cpu").strip().lower(),
            max_input_len=int(os.getenv("SAGE_VALIDATOR_MAX_INPUT_LEN") or "512"),
            warmup=_env_bool("SAGE_VALIDATOR_WARMUP", True),
        )


# ───────────────────────────────────────────────────────────────────────
# Validator
# ───────────────────────────────────────────────────────────────────────

# Verdict для одной правки
VERDICT_AGREE = "agree"          # sage воспроизвёл `after`
VERDICT_DISAGREE = "disagree"    # sage оставил `before` (не стал править)
VERDICT_UNKNOWN = "unknown"      # sage сделал что-то третье


class SageValidator:
    """Lazy-loaded wrapper над sage-fredt5 для GEC-валидации.

    Использование:
        validator = SageValidator(SageConfig.from_env())
        if validator.is_available():
            sage_text = validator.correct(raw_text)
            verdict = validator.judge(before, after, sage_text)
    """

    def __init__(self, config: SageConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._loaded = False

    # ───── load / availability ─────

    def is_available(self) -> bool:
        """True если sage готов отвечать. Lazy-загружает при первом вызове."""
        if not self.config.enabled:
            return False
        if self._loaded:
            return self._model is not None
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            self._loaded = True  # помечаем сразу — повторных попыток не делаем
            try:
                # Lazy imports — без torch+transformers сервер должен работать
                from transformers import (  # type: ignore
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                logger.info(
                    "Sage: загружаю модель %s на %s ...",
                    self.config.model_name, self.config.device,
                )
                tok = AutoTokenizer.from_pretrained(self.config.model_name)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
                mdl = mdl.to(self.config.device)
                mdl.eval()
                self._tokenizer = tok
                self._model = mdl
                logger.info(
                    "Sage: модель загружена (params≈%dM)",
                    sum(p.numel() for p in mdl.parameters()) // 1_000_000,
                )
                if self.config.warmup:
                    try:
                        self.correct("Привет, мир.")
                        logger.info("Sage: warmup-проход ОК")
                    except Exception as e:
                        logger.warning("Sage: warmup упал: %s", e)
            except ImportError as e:
                self._load_error = f"transformers/torch не установлены: {e}"
                logger.warning(
                    "Sage отключён: %s. "
                    "Установите `pip install transformers torch` для включения.",
                    self._load_error,
                )
            except Exception as e:  # noqa: BLE001
                self._load_error = f"{type(e).__name__}: {e}"
                logger.error("Sage: не удалось загрузить модель: %s", self._load_error)

    # ───── inference ─────

    def correct(self, text: str) -> str:
        """Прогоняет text через sage и возвращает скорректированную версию.

        Если sage недоступен — возвращает text без изменений.
        """
        if not self.is_available():
            return text
        try:
            import torch  # type: ignore

            assert self._model is not None and self._tokenizer is not None
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_len,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_length=self.config.max_input_len,
                    num_beams=1,                 # greedy — детерминированно и быстро
                    do_sample=False,
                    early_stopping=True,
                )
            decoded = self._tokenizer.decode(out[0], skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Sage: inference упал: %s", e)
            return text

    # ───── verdict ─────

    def judge(self, before: str, after: str, sage_text: str) -> str:
        """Сравнивает (before, after) с sage_text и возвращает VERDICT_*.

        Логика:
          - `after` встречается в sage_text  →  AGREE
          - `before` встречается в sage_text →  DISAGREE
                                                (sage не стал править)
          - иначе                            →  UNKNOWN

        Сравнение по подстроке case-sensitive (русский язык чувствителен
        к регистру в названиях). Для устойчивости пунктуации применяется
        нормализация whitespace через _normalize_ws().
        """
        if not before or not after or not sage_text:
            return VERDICT_UNKNOWN
        norm_before = _normalize_ws(before)
        norm_after = _normalize_ws(after)
        norm_sage = _normalize_ws(sage_text)
        # Sage может изменить case первой буквы — пробуем оба варианта.
        after_in = (norm_after in norm_sage) or (
            _swap_first_case(norm_after) in norm_sage
        )
        if after_in:
            return VERDICT_AGREE
        before_in = (norm_before in norm_sage) or (
            _swap_first_case(norm_before) in norm_sage
        )
        if before_in:
            return VERDICT_DISAGREE
        return VERDICT_UNKNOWN

    def category_matches(self, category: str) -> bool:
        """True если category-строка пункта CHANGES попадает под фильтр.

        Категории конфигурируются `SAGE_VALIDATOR_CATEGORIES` (substring,
        case-insensitive). Пустой список = разрешены все категории.

        Sage обучена на орфографических ошибках; для согласования/управления
        она ненадёжна (см. ЖУРНАЛ_v1.6.md PoC). Поэтому по умолчанию
        фильтр применяется ТОЛЬКО к пунктам с категорией «орфография».
        """
        if not self.config.categories:
            return True
        cat_lower = (category or "").lower()
        return any(c in cat_lower for c in self.config.categories)

    def should_drop(self, verdict: str, *, category: str = "") -> bool:
        """По текущему domain/mode/category решает, нужно ли дропать правку.

        В dryrun-режиме всегда False (только логируем verdict, не дропаем).
        В enforce-режиме:
          - категория ДОЛЖНА попадать под `SAGE_VALIDATOR_CATEGORIES`,
            иначе False (sage может ошибаться на не-орфографии);
          - DISAGREE → True всегда (sage явно против правки);
          - UNKNOWN → True только в general (admin сохраняет recall).
        """
        if self.config.mode == "dryrun":
            return False
        if not self.category_matches(category):
            return False
        if verdict == VERDICT_DISAGREE:
            return True
        if verdict == VERDICT_UNKNOWN and self.config.domain == "general":
            return True
        # AGREE — никогда не дропаем;
        # UNKNOWN в admin-режиме — не дропаем (recall важнее).
        return False


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    """Сводит любые whitespace-последовательности к одному пробелу, обрезает края."""
    return _WS_RE.sub(" ", s).strip()


def _swap_first_case(s: str) -> str:
    """Меняет регистр ПЕРВОЙ буквы — больше ничего.

    Sage иногда возвращает «Проверочное» когда T-lite отдал «проверочное»
    (или наоборот) — для целей VERDICT эта разница не должна влиять.
    """
    if not s:
        return s
    first = s[0]
    if first.isupper():
        return first.lower() + s[1:]
    if first.islower():
        return first.upper() + s[1:]
    return s


# ───────────────────────────────────────────────────────────────────────
# Module-level singleton (для FastAPI dependency injection)
# ───────────────────────────────────────────────────────────────────────

_singleton: Optional[SageValidator] = None
_singleton_lock = threading.Lock()


def get_validator() -> SageValidator:
    """Возвращает singleton SageValidator (lazy)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SageValidator(SageConfig.from_env())
    return _singleton


def reset_validator_for_testing() -> None:
    """Сбрасывает singleton — только для тестов."""
    global _singleton
    with _singleton_lock:
        _singleton = None
