"""Общая пост-обработка ответов LLM (Ollama / OpenRouter).

Этот модуль извлечён из `server/local/main.py` (v2.1), чтобы облачный
сервер (`server/cloud/main.py`) мог использовать тот же pipeline без
дублирования кода. Функции «pure»-уровня (зависят только от текста)
не нуждаются в дополнительных параметрах. Функции, зависящие от
опциональных сервисов (морф-фильтр, морф-детектор, пользовательский
словарь, LanguageTool, sage-валидатор), принимают сервис аргументом;
если сервис `None`, функция работает как no-op.

Все логи идут через `logger = logging.getLogger(__name__)`. Локальный
и облачный сервер инициализируют общий логгер через `setup_logger()`
до первого вызова — настройка имени корня (`ai_suggester.*`) применит
ротацию и формат к этому модулю автоматически.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Optional


logger = logging.getLogger(__name__)


# ─── Регулярки и литералы, разделяемые всеми хелперами ────────────────

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Угловые/типографские кавычки, встречающиеся в ===CHANGES===. Одиночные
# ' и ` не включаем — они ложно срабатывают на апостроф/транслитерацию.
_QUOTE_CHARS = "«»\"“”‘’‚‛„"

# Разделитель между «было» и «стало». Допускаем стрелки (→, ->), тире
# (—, –, -) и текстовые связки, включая обороты «заменено/исправлено на».
_CHANGE_PAIR_RE = re.compile(
    rf"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]+)[{_QUOTE_CHARS}]"
    rf"[^{_QUOTE_CHARS}]*?"
    rf"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]+)[{_QUOTE_CHARS}]",
    re.IGNORECASE,
)

_LEADING_NUM_RE = re.compile(r"^\s*\d+\.\s*")
_QUOTED_GREEDY_RE = re.compile(
    rf"^\s*[{_QUOTE_CHARS}](.+)[{_QUOTE_CHARS}]\s*$"
)
_CHANGE_NUM_RE = re.compile(r"^(\s*)(\d+)\.(\s*)(.*)$")


# ─── Pure helpers (только текст на входе) ─────────────────────────────


def _strip_thinking(text: str) -> str:
    """Срезает <think>…</think> и leading-рассуждения, если модель проигнорировала /no_think.

    Возвращает «чистый» ответ. Если в тексте нет ни тегов <think>, ни маркера
    ===CORRECTED===, не трогаем — пусть верхний слой сам разбирается.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    idx = cleaned.find("===CORRECTED===")
    if idx > 0:
        cleaned = cleaned[idx:]
    return cleaned.strip()


def _parse_change_pair_robust(line: str) -> Optional[tuple[str, str]]:
    """Извлекает (before, after) из строки CHANGES, корректно обрабатывая
    вложенные кавычки. v1.8.2: _CHANGE_PAIR_RE использует non-greedy
    срез между [^кавычек]+, и на строке вида «адм…здания «ЦСН ВО»» →
    «адм…здания «ЦСН ВО»» он матчится на внутренней «ЦСН ВО», а не на
    внешней парe — в итоге before/after не равны и фильтр пропускает
    идемпотентный пункт.

    Эта функция работает по структуре строки:
      `N. «before» → «after» | explanation` → (before, after).
    """
    s = _LEADING_NUM_RE.sub("", line.strip())
    if " | " in s:
        s = s.split(" | ", 1)[0]
    for sep in (" → ", " -> "):
        if sep in s:
            left, right = s.rsplit(sep, 1)
            lm = _QUOTED_GREEDY_RE.match(left.strip())
            rm = _QUOTED_GREEDY_RE.match(right.strip())
            if lm and rm:
                return (lm.group(1), rm.group(1))
            return None
    return None


def _drop_idempotent_changes(text: str) -> str:
    """Удаляет из блока ===CHANGES=== пункты вида «X → X»."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    kept: list[str] = []
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        if pair and pair[0].strip() == pair[1].strip():
            logger.debug("Фильтрую идемпотентный пункт: %s", line.strip())
            continue
        if pair and ("…" in pair[0] or "..." in pair[0]
                     or "…" in pair[1] or "..." in pair[1]):
            logger.info("Фильтрую пункт с многоточием в цитате: %s", line.strip())
            continue
        kept.append(line)

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _renumber_changes(text: str) -> str:
    """v1.7.3: пере-нумеровывает пункты ===CHANGES=== подряд."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    new_lines: list[str] = []
    next_num = 1
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        m = _CHANGE_NUM_RE.match(line)
        if m:
            indent, _old, sep, content = m.group(1), m.group(2), m.group(3), m.group(4)
            new_lines.append(f"{indent}{next_num}.{sep}{content}")
            next_num += 1
        else:
            new_lines.append(line)
    new_changes = "\n".join(new_lines).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _replace_corrected_body(text: str, new_body: str) -> str:
    """Заменяет содержимое блока ===CORRECTED===…===CHANGES=== на new_body."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    try:
        before, rest = text.split("===CORRECTED===", 1)
        _, tail = rest.split("===CHANGES===", 1)
    except ValueError:
        return text
    return f"{before}===CORRECTED===\n{new_body.strip()}\n===CHANGES==={tail}"


def _drop_changes_not_in_text(text: str, raw_text: str) -> str:
    """Дропает пункты ===CHANGES===, чьё «было» не является подстрокой raw_text."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    kept: list[str] = []
    dropped_count = 0
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            if not quote_before or quote_before not in raw_text:
                logger.info(
                    "Дроп пункта: «%s» нет в raw_text (галлюцинация): %s",
                    quote_before, line.strip(),
                )
                dropped_count += 1
                continue
        kept.append(line)

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    if dropped_count:
        logger.info("Отфильтровано %d пункт(ов) с галлюцинированным «было»", dropped_count)

    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _is_eyo_only_substitution(before: str, after: str) -> bool:
    """True, если before/after отличаются ИСКЛЮЧИТЕЛЬНО подменой ё↔е (Ё↔Е)."""
    if not before or not after or before == after:
        return False
    norm_before = before.replace("ё", "е").replace("Ё", "Е")
    norm_after = after.replace("ё", "е").replace("Ё", "Е")
    return norm_before == norm_after


def _drop_eyo_substitutions(text: str, raw_text: str) -> str:
    """Дропает пункты ===CHANGES===, отличающиеся только заменой ё↔е,
    и откатывает подмену в ===CORRECTED==="""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    if "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        head, rest1 = text.split("===CORRECTED===", 1)
        corrected_block, rest2 = rest1.split("===CHANGES===", 1)
        changes_block, tail = rest2.split("===END===", 1)
    except ValueError:
        return text

    new_corrected = corrected_block
    kept: list[str] = []
    dropped_count = 0
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            quote_after = m.group(2).strip()
            if (
                _is_eyo_only_substitution(quote_before, quote_after)
                and quote_before in raw_text
                and quote_after in new_corrected
            ):
                new_corrected = new_corrected.replace(quote_after, quote_before)
                logger.info(
                    "Дроп стилистической ё-замены: «%s» → «%s» (откат в CORRECTED)",
                    quote_before, quote_after,
                )
                dropped_count += 1
                continue
        kept.append(line)

    if dropped_count == 0:
        return text

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    logger.info("Отфильтровано %d пункт(ов) ё-замены (стилистика, не ошибка)", dropped_count)

    new_changes = "\n".join(kept).rstrip() + "\n"
    return (
        f"{head}===CORRECTED==={new_corrected}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )


def _undo_eyo_in_text(corrected: str, raw_text: str) -> str:
    """Посимвольно откатывает в `corrected` подмены ё→е (Ё→Е), которых нет в raw_text."""
    if "ё" not in corrected and "Ё" not in corrected:
        return corrected
    if not raw_text:
        return corrected
    matcher = difflib.SequenceMatcher(None, raw_text, corrected, autojunk=False)
    parts: list[str] = []
    undone = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(corrected[j1:j2])
        elif tag == "replace":
            raw_seg = raw_text[i1:i2]
            corr_seg = corrected[j1:j2]
            if len(raw_seg) == len(corr_seg):
                fixed_chars: list[str] = []
                for r_ch, c_ch in zip(raw_seg, corr_seg):
                    if c_ch == "ё" and r_ch == "е":
                        fixed_chars.append("е")
                        undone += 1
                    elif c_ch == "Ё" and r_ch == "Е":
                        fixed_chars.append("Е")
                        undone += 1
                    else:
                        fixed_chars.append(c_ch)
                parts.append("".join(fixed_chars))
            else:
                parts.append(corr_seg)
        elif tag == "insert":
            parts.append(corrected[j1:j2])
    if undone == 0:
        return corrected
    logger.info("Откат ё→е в CORRECTED: %d символ(ов) восстановлено по raw_text", undone)
    return "".join(parts)


def _undo_eyo_in_corrected_block(text: str, raw_text: str) -> str:
    """Применяет `_undo_eyo_in_text` к содержимому ===CORRECTED===."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    try:
        head, rest = text.split("===CORRECTED===", 1)
        corrected_block, tail = rest.split("===CHANGES===", 1)
    except ValueError:
        return text
    new_corrected = _undo_eyo_in_text(corrected_block, raw_text)
    if new_corrected == corrected_block:
        return text
    return f"{head}===CORRECTED==={new_corrected}===CHANGES==={tail}"


def _expand_word_context(s: str, lo: int, hi: int) -> tuple[int, int]:
    """Расширяет [lo, hi) до границ слов с прихватом одного соседнего слова."""
    while lo > 0 and not s[lo - 1].isspace():
        lo -= 1
    if lo > 0:
        while lo > 0 and s[lo - 1].isspace():
            lo -= 1
        while lo > 0 and not s[lo - 1].isspace():
            lo -= 1
    while hi < len(s) and not s[hi].isspace():
        hi += 1
    if hi < len(s):
        while hi < len(s) and s[hi].isspace():
            hi += 1
        while hi < len(s) and not s[hi].isspace():
            hi += 1
    return lo, hi


def _rebuild_changes_from_diff(raw_text: str, corrected_text: str) -> list[str]:
    """Восстанавливает пункты CHANGES из посимвольного diff между raw_text и corrected_text."""
    if not raw_text or not corrected_text or raw_text == corrected_text:
        return []
    sm = difflib.SequenceMatcher(None, raw_text, corrected_text, autojunk=False)
    entries: list[str] = []
    seen_before: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a, b = _expand_word_context(raw_text, i1, i2)
        c, d = _expand_word_context(corrected_text, j1, j2)
        before_part = raw_text[a:b].strip()
        after_part = corrected_text[c:d].strip()
        if not before_part or before_part == after_part:
            continue
        if before_part not in raw_text:
            continue
        if before_part in seen_before:
            continue
        seen_before.add(before_part)
        entries.append(
            f"«{before_part}» → «{after_part}» | автоправка по diff "
            f"(модель не указала точную причину)"
        )
    return entries


def _has_real_change_items(text: str) -> bool:
    """True, если в CHANGES есть хотя бы один содержательный пункт."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return False
    try:
        _, rest = text.split("===CHANGES===", 1)
        block, _ = rest.split("===END===", 1)
    except ValueError:
        return False
    for line in block.splitlines():
        if not line.strip():
            continue
        if "Ошибок не найдено" in line:
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m and m.group(1).strip() and m.group(1).strip() != m.group(2).strip():
            return True
    return False


def _had_any_change_pairs(text: str) -> bool:
    """Был ли в CHANGES хотя бы один пункт «X» → «Y» ДО фильтрации."""
    if "===CHANGES===" not in text:
        return False
    try:
        _, rest = text.split("===CHANGES===", 1)
    except ValueError:
        return False
    block = rest.split("===END===", 1)[0] if "===END===" in rest else rest
    for line in block.splitlines():
        if _CHANGE_PAIR_RE.search(line):
            return True
    return False


def _extract_corrected_body(text: str) -> str:
    """Возвращает содержимое блока CORRECTED без переносов в начале/конце."""
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return ""
    try:
        _, after = text.split("===CORRECTED===", 1)
        body, _ = after.split("===CHANGES===", 1)
    except ValueError:
        return ""
    return body.strip()


def _replace_changes_block(text: str, entries: list[str]) -> str:
    """Заменяет содержимое блока CHANGES на пронумерованный список entries."""
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before, rest = text.split("===CHANGES===", 1)
        _, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    if entries:
        new_block = "\n".join(f"{i}. {e}" for i, e in enumerate(entries, start=1))
    else:
        new_block = "1. Ошибок не найдено. Текст соответствует нормам."
    return f"{before}===CHANGES===\n{new_block}\n===END==={tail}"


def _complete_changes_from_corrected(text: str, raw_text: str) -> str:
    """v1.8.4: закрывает рассинхрон CHANGES↔CORRECTED через diff(simulated → corrected_body)."""
    if not text or not raw_text:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text

    corrected_body = _extract_corrected_body(text)
    if not corrected_body or corrected_body == raw_text:
        return text

    try:
        head, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text

    existing_pairs: list[tuple[str, str]] = []
    for raw_line in changes_block.splitlines():
        line = raw_line.strip()
        if not line or "Ошибок не найдено" in line:
            continue
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        if pair is None:
            continue
        before_q, after_q = pair[0].strip(), pair[1].strip()
        if not before_q or not after_q or before_q == after_q:
            continue
        existing_pairs.append((before_q, after_q))

    simulated = raw_text
    for before_q, after_q in existing_pairs:
        idx = simulated.find(before_q)
        if idx < 0:
            continue
        simulated = simulated[:idx] + after_q + simulated[idx + len(before_q):]

    if simulated.strip() == corrected_body.strip():
        return text

    missing_entries = _rebuild_changes_from_diff(simulated, corrected_body)
    if not missing_entries:
        return text

    existing_befores = {b.strip().lower() for b, _ in existing_pairs}
    new_entries: list[str] = []
    for entry in missing_entries:
        m = _CHANGE_PAIR_RE.search(entry)
        if m is None:
            continue
        b_norm = m.group(1).strip().lower()
        if b_norm in existing_befores:
            continue
        new_entries.append(entry)
        existing_befores.add(b_norm)

    if not new_entries:
        return text

    logger.info(
        "v1.8.4: добавлено %d пункт(ов) CHANGES для синхронизации с CORRECTED",
        len(new_entries),
    )

    suffix = "\n".join(new_entries)
    existing_kept = changes_block.rstrip("\n")
    if existing_kept.strip() and "Ошибок не найдено" in existing_kept:
        existing_kept = ""
    if existing_kept:
        new_changes = existing_kept + "\n" + suffix + "\n"
    else:
        new_changes = "\n" + suffix + "\n"
    return f"{head}===CHANGES==={new_changes}===END==={tail}"


# ─── Service-injecting helpers (need optional dependency) ─────────────


def _drop_user_dict_changes(text: str, user_dict) -> str:  # noqa: ANN001
    """Дропает CHANGES-пункты, в которых модель пытается «исправить»
    whitelisted-термин. `user_dict` может быть None — тогда no-op.
    """
    if user_dict is None:
        return text
    words = user_dict.list_words()
    if not words:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    word_patterns = [re.escape(w) for w in words]
    word_re = re.compile(
        r"(?<![\w-])(?:" + "|".join(word_patterns) + r")(?![\w-])",
        re.IGNORECASE,
    )
    kept: list[str] = []
    dropped = 0
    for line in changes_block.splitlines():
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            before_q, after_q = m.group(1), m.group(2)
            if word_re.search(before_q) or word_re.search(after_q):
                logger.info(
                    "Дроп правки whitelisted-термина (user_dict): «%s» → «%s»",
                    before_q.strip(), after_q.strip(),
                )
                dropped += 1
                continue
        kept.append(line)
    if dropped == 0:
        return text
    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]
    new_changes = "\n".join(kept).rstrip() + "\n"
    return f"{before_block}===CHANGES===\n{new_changes.lstrip()}===END==={tail}"


def _drop_morph_case_substitutions(text: str, raw_text: str, morph_filter) -> str:  # noqa: ANN001
    """v1.7: дроп падежных «улучшений» через pymorphy3 morph-filter.
    `morph_filter` может быть None или с `available=False` — тогда no-op.
    """
    if morph_filter is None or not getattr(morph_filter, "available", False):
        return text
    if "===CORRECTED===" not in text or "===CHANGES===" not in text:
        return text
    if "===END===" not in text:
        return text
    if not raw_text:
        return text
    try:
        head, rest1 = text.split("===CORRECTED===", 1)
        corrected_block, rest2 = rest1.split("===CHANGES===", 1)
        changes_block, tail = rest2.split("===END===", 1)
    except ValueError:
        return text

    new_corrected = corrected_block
    kept: list[str] = []
    dropped_count = 0
    reverted_in_corrected = 0
    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            kept.append(line)
            continue
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            quote_before = m.group(1).strip()
            quote_after = m.group(2).strip()
            if (
                morph_filter.is_hallucinated_case_change(quote_before, quote_after, raw_text)
                and quote_after in new_corrected
            ):
                new_corrected = new_corrected.replace(quote_after, quote_before)
                logger.info(
                    "Дроп падежной «улучшалки»: «%s» → «%s» "
                    "(одна лемма, разный падеж, нет управляющего предлога; откат в CORRECTED)",
                    quote_before, quote_after,
                )
                dropped_count += 1
                continue
            pairs = morph_filter.find_hallucinated_pairs_in_compound(
                quote_before, quote_after, raw_text
            )
            if pairs:
                for bw, aw in pairs:
                    if aw in new_corrected:
                        new_corrected = new_corrected.replace(aw, bw)
                        reverted_in_corrected += 1
                        logger.info(
                            "Дроп падежной «улучшалки» (compound): «%s» → «%s» "
                            "(внутри компаунда «%s» → «%s»; откат в CORRECTED)",
                            bw, aw, quote_before[:60], quote_after[:60],
                        )
                if morph_filter.is_compound_fully_hallucinated(
                    quote_before, quote_after, raw_text
                ):
                    dropped_count += 1
                    continue
        kept.append(line)

    if dropped_count == 0 and reverted_in_corrected == 0:
        return text

    non_empty = [ln for ln in kept if re.search(r"\w", ln)]
    has_real_item = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real_item:
        kept = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    if dropped_count > 0:
        logger.info(
            "Отфильтровано %d пункт(ов) падежных «улучшений» (pymorphy3 morph-filter)",
            dropped_count,
        )
    if reverted_in_corrected > 0:
        logger.info(
            "Откат %d compound-словоподмен в CORRECTED (pymorphy3 morph-filter)",
            reverted_in_corrected,
        )

    new_changes = "\n".join(kept).rstrip() + "\n"
    return (
        f"{head}===CORRECTED==={new_corrected}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )


def _enrich_changes_with_detector(text: str, raw_text: str, morph_detector, user_dict=None) -> str:  # noqa: ANN001
    """v1.8a: Обогащает CHANGES пунктами morph-детектора."""
    if morph_detector is None or not getattr(morph_detector, "available", False):
        return text
    if not raw_text:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text
    whitelist = user_dict.as_frozenset() if user_dict is not None else None
    try:
        errors = morph_detector.detect_errors(raw_text, whitelist=whitelist)
    except Exception as exc:
        logger.warning("MorphDetector упал на raw_text (size=%d): %s", len(raw_text), exc)
        return text
    if not errors:
        return text
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    existing_befores: set[str] = set()
    for line in changes_block.splitlines():
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            existing_befores.add(m.group(1).strip().lower())
    corrected_body = _extract_corrected_body(text) or raw_text
    new_lines: list[str] = []
    added_pairs = 0
    patched_corrected = corrected_body
    for err in errors:
        if err.before.strip().lower() in existing_befores:
            continue
        if err.before.strip().lower() in {
            e.split("«")[1].split("»")[0].lower() for e in new_lines if "«" in e
        }:
            continue
        if not err.suggestion or err.suggestion == err.before:
            logger.info(
                "MorphDetector: OOV-слово «%s» (offset=%d) — не добавляем в CHANGES",
                err.before, err.offset,
            )
            continue
        new_lines.append(err.to_change_line(0))
        existing_befores.add(err.before.strip().lower())
        added_pairs += 1
        if err.before in patched_corrected:
            patched_corrected = patched_corrected.replace(err.before, err.suggestion, 1)
    if added_pairs == 0:
        return text
    existing_kept = changes_block.rstrip()
    if existing_kept and "Ошибок не найдено" in existing_kept:
        existing_kept = ""
    if existing_kept:
        merged_changes = existing_kept + "\n" + "\n".join(new_lines) + "\n"
    else:
        merged_changes = "\n" + "\n".join(new_lines) + "\n"
    logger.info(
        "MorphDetector: добавлено %d пункт(ов) в CHANGES (пропущено моделью)",
        added_pairs,
    )
    new_text = before_block + "===CHANGES===" + merged_changes + "===END===" + tail
    if patched_corrected != corrected_body:
        new_text = _replace_corrected_body(new_text, patched_corrected)
    return new_text


def _enrich_changes_with_languagetool(text: str, raw_text: str, lt_client, user_dict=None) -> str:  # noqa: ANN001
    """v2.0-b: добавляет стилистические/типографские правки от LanguageTool."""
    if lt_client is None or not getattr(lt_client, "available", False):
        return text
    if not raw_text:
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text
    try:
        matches = lt_client.check(raw_text)
    except Exception as exc:
        logger.warning("LanguageTool упал на raw_text (size=%d): %s", len(raw_text), exc)
        return text
    if not matches:
        return text
    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
    except ValueError:
        return text
    existing_befores: set[str] = set()
    for line in changes_block.splitlines():
        m = _CHANGE_PAIR_RE.search(line)
        if m:
            existing_befores.add(m.group(1).strip().lower())
    corrected_body = _extract_corrected_body(text) or raw_text
    new_lines: list[str] = []
    added_pairs = 0
    patched_corrected = corrected_body
    whitelist = (
        user_dict.as_frozenset() if user_dict is not None else frozenset()
    )
    for lt in matches:
        if not lt.suggestion or lt.suggestion == lt.before:
            continue
        before_key = lt.before.strip().lower()
        if before_key in existing_befores:
            continue
        if lt.before in whitelist or before_key in whitelist:
            continue
        new_lines.append(lt.to_change_line(0))
        existing_befores.add(before_key)
        added_pairs += 1
        if lt.before in patched_corrected:
            patched_corrected = patched_corrected.replace(lt.before, lt.suggestion, 1)
    if added_pairs == 0:
        return text
    existing_kept = changes_block.rstrip()
    if existing_kept and "Ошибок не найдено" in existing_kept:
        existing_kept = ""
    if existing_kept:
        merged_changes = existing_kept + "\n" + "\n".join(new_lines) + "\n"
    else:
        merged_changes = "\n" + "\n".join(new_lines) + "\n"
    logger.info(
        "LanguageTool: добавлено %d пункт(ов) в CHANGES (стиль/типографика)",
        added_pairs,
    )
    new_text = before_block + "===CHANGES===" + merged_changes + "===END===" + tail
    if patched_corrected != corrected_body:
        new_text = _replace_corrected_body(new_text, patched_corrected)
    return new_text


def _filter_changes_with_sage(text: str, raw_text: str, sage_validator) -> str:  # noqa: ANN001
    """v1.8c: пост-валидация CHANGES через sage-fredt5-distilled-95m.
    `sage_validator` может быть None — тогда no-op.
    """
    if sage_validator is None or not sage_validator.is_available():
        return text
    if not raw_text or not raw_text.strip():
        return text
    if "===CHANGES===" not in text or "===END===" not in text:
        return text
    if "===CORRECTED===" not in text:
        return text

    try:
        sage_text = sage_validator.correct(raw_text)
    except Exception as e:
        logger.warning("Sage: ошибка correct(), пропускаю валидацию: %s", e)
        return text
    if not sage_text or sage_text == raw_text:
        return text

    try:
        before_block, rest = text.split("===CHANGES===", 1)
        changes_block, tail = rest.split("===END===", 1)
        corrected_start = before_block.find("===CORRECTED===")
        if corrected_start < 0:
            return text
        head = before_block[:corrected_start]
        corrected_full = before_block[corrected_start:]
        corrected_body_match = re.match(
            r"===CORRECTED===\s*\n?(.*)\Z", corrected_full, flags=re.DOTALL
        )
        if corrected_body_match is None:
            return text
        corrected_body = corrected_body_match.group(1)
    except ValueError:
        return text

    kept_lines: list[str] = []
    dropped = 0
    new_corrected = corrected_body

    for raw_line in changes_block.splitlines():
        line = raw_line.rstrip()
        pair = _parse_change_pair_robust(line)
        if pair is None:
            m = _CHANGE_PAIR_RE.search(line)
            if m:
                pair = (m.group(1), m.group(2))
        if pair is None:
            kept_lines.append(line)
            continue

        before_q, after_q = pair[0], pair[1]
        category = ""
        if "|" in line:
            category = line.split("|", 1)[1].strip()
        verdict = sage_validator.judge(before_q, after_q, sage_text)
        logger.info(
            "Sage[%s/%s/cat=%r]: verdict=%s для %r→%r",
            sage_validator.config.mode, sage_validator.config.domain,
            category[:40], verdict, before_q, after_q,
        )
        if sage_validator.should_drop(verdict, category=category):
            logger.info(
                "Sage[enforce]: ДРОП правки %r→%r (verdict=%s, cat=%r)",
                before_q, after_q, verdict, category[:40],
            )
            dropped += 1
            idx = new_corrected.find(after_q)
            if idx >= 0:
                new_corrected = (
                    new_corrected[:idx] + before_q + new_corrected[idx + len(after_q):]
                )
            continue
        kept_lines.append(line)

    if dropped == 0:
        return text

    non_empty = [ln for ln in kept_lines if re.search(r"\w", ln)]
    has_real = any(re.match(r"\s*\d+\.\s*\S", ln) for ln in non_empty)
    if not has_real:
        kept_lines = ["", "1. Ошибок не найдено. Текст соответствует нормам.", ""]

    new_changes = "\n".join(kept_lines).rstrip() + "\n"
    new_corrected_block = "===CORRECTED===\n" + new_corrected.lstrip("\n")
    if not new_corrected_block.endswith("\n"):
        new_corrected_block += "\n"
    return (
        f"{head}{new_corrected_block}"
        f"===CHANGES===\n{new_changes.lstrip()}===END==={tail}"
    )
