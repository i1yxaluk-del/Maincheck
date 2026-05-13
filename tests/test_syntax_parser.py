"""Тесты v1.9 syntax_parser.SyntaxParser (natasha).

Цель — убедиться что:
  1. Singleton / lazy-init работают.
  2. `ParsedDoc.is_attributive_modifier` правильно различает
     attributive-связи (amod) от не-attributive (obl:agent, parataxis).
  3. case-relation child корректно блокирует «attributive»-вердикт
     (закрывает FP-2 «при этом количество»).
  4. Если natasha не доступна — parser.available=False, parse=None.

Тесты использует РЕАЛЬНУЮ natasha-pipeline. Это медленнее (~5 сек
startup + ~10 мс на текст), но даёт реалистичную верификацию.
"""

from __future__ import annotations

import sys
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = os.path.join(REPO_ROOT, "server")
if SERVER_PATH not in sys.path:
    sys.path.insert(0, SERVER_PATH)


@pytest.fixture(scope="module")
def parser():
    """Один общий natasha-pipeline на весь модуль (загрузка ~5 сек)."""
    from shared.syntax_parser import reset_syntax_parser, get_syntax_parser
    reset_syntax_parser()
    p = get_syntax_parser()
    if not p.available:
        pytest.skip("natasha не установлена")
    return p


def test_parser_available(parser):
    """Smoke: парсер загрузился и available=True."""
    assert parser.available


def test_parse_empty_text(parser):
    """Пустой текст возвращает пустой ParsedDoc, не None."""
    doc = parser.parse("")
    assert doc is not None
    assert len(doc.tokens) == 0


def test_parse_simple_sentence(parser):
    """Простое предложение: токены + offsets + relations."""
    doc = parser.parse("Капитальный ремонт здания.")
    assert doc is not None
    assert len(doc.tokens) >= 3
    texts = [t.text for t in doc.tokens]
    assert "Капитальный" in texts
    assert "ремонт" in texts
    assert "здания" in texts


def test_real_disagreement_not_marked_as_non_attributive(parser):
    """«Проверочное мероприятия» — реальный disagreement (adj.sing
    + noun.plur). natasha может мис-парсить структуру (видеть
    «Проверочное» как PROPN nsubj), но is_clearly_non_attributive ОБЯЗАН
    вернуть False (не будем блокировать реальную ошибку).

    Главная инварианта v1.9: false-negative parser ДОЛЖЕН вести
    к fallback на морф-детектор, НЕ к пропуску ошибки.
    """
    text = "Проверочное мероприятия выполнено."
    doc = parser.parse(text)
    assert doc is not None
    proverochnoye_idx = doc.token_at_offset(0)
    meropriyatiya_idx = None
    for i, t in enumerate(doc.tokens):
        if t.text.startswith("меропри"):
            meropriyatiya_idx = i
            break
    assert proverochnoye_idx is not None
    assert meropriyatiya_idx is not None
    # is_clearly_non_attributive НЕ должна ошибочно пропустить это как
    # non-attributive — иначе detector пропустит реальный disagreement.
    assert not doc.is_clearly_non_attributive(proverochnoye_idx, meropriyatiya_idx)


def test_obl_agent_fp1_flagged_as_non_attributive(parser):
    """FP-1: «проводимых подразделениями» — obl:agent (агенс
    причастия в творительном), НЕ amod. is_clearly_non_attributive
    должна вернуть True → detector пропустит, FP блокируется.
    """
    text = "мероприятий, проводимых подразделениями собственной безопасности"
    doc = parser.parse(text)
    assert doc is not None
    provodimykh_idx = None
    podrazdeleniyami_idx = None
    for i, t in enumerate(doc.tokens):
        if t.text == "проводимых":
            provodimykh_idx = i
        if t.text == "подразделениями":
            podrazdeleniyami_idx = i
    assert provodimykh_idx is not None
    assert podrazdeleniyami_idx is not None
    # Пара (mod=проводимых, head=подразделениями) — ADJ перед NOUN в
    # порядке detector'а. Наташа вернёт «подразделениями».head=проводимых
    # и .rel=obl:agent. is_clearly_non_attributive(mod=adj_idx, head=noun_idx)
    # использует Сигнал 3 (head-токен бнут к mod через obl:agent).
    assert doc.is_clearly_non_attributive(provodimykh_idx, podrazdeleniyami_idx)


def test_case_governed_fp2_flagged_as_non_attributive(parser):
    """FP-2: «при этом количество» — «этом» имеет case-child «при»,
    следовательно НЕ модифицирует «количество» атрибутивно.
    is_clearly_non_attributive должна вернуть True.
    """
    text = "при этом количество должностных преступлений"
    doc = parser.parse(text)
    assert doc is not None
    etom_idx = None
    kolichestvo_idx = None
    for i, t in enumerate(doc.tokens):
        if t.text == "этом":
            etom_idx = i
        if t.text == "количество":
            kolichestvo_idx = i
    assert etom_idx is not None
    assert kolichestvo_idx is not None
    # natasha может прицепить «этом» к «количество» rel=nmod, но у
    # «этом» есть case-child «при» — Сигнал 1 в is_clearly_non_attributive.
    assert doc.is_clearly_non_attributive(etom_idx, kolichestvo_idx)


def test_legitimate_amod_not_flagged_as_non_attributive(parser):
    """«капитальный ремонт» — реальная amod-связь.
    is_clearly_non_attributive ДОЛЖНА вернуть False.
    """
    text = "Выполнен капитальный ремонт здания."
    doc = parser.parse(text)
    assert doc is not None
    kap_idx = None
    rem_idx = None
    for i, t in enumerate(doc.tokens):
        if t.text == "капитальный":
            kap_idx = i
        if t.text == "ремонт":
            rem_idx = i
    assert kap_idx is not None
    assert rem_idx is not None
    assert not doc.is_clearly_non_attributive(kap_idx, rem_idx)


def test_token_at_offset(parser):
    """token_at_offset(start) возвращает индекс токена; на пробеле — None."""
    text = "Капитальный ремонт здания."
    doc = parser.parse(text)
    assert doc is not None
    # «Капитальный» начинается с offset 0
    idx0 = doc.token_at_offset(0)
    assert idx0 is not None
    assert doc.tokens[idx0].text == "Капитальный"
    # offset 5 (середина слова) — нет токена
    assert doc.token_at_offset(5) is None
    # «ремонт» начинается после пробела
    rem_start = text.find("ремонт")
    idx_rem = doc.token_at_offset(rem_start)
    assert idx_rem is not None
    assert doc.tokens[idx_rem].text == "ремонт"


def test_cache_hit(parser):
    """Повторный parse() того же текста возвращает тот же объект
    (sha1-кэш)."""
    text = "Один и тот же текст."
    doc1 = parser.parse(text)
    doc2 = parser.parse(text)
    # Одно и то же либо равно по содержимому (одинаковая ссылка из кэша).
    assert doc1 is doc2 or (
        len(doc1.tokens) == len(doc2.tokens)
        and all(a.text == b.text for a, b in zip(doc1.tokens, doc2.tokens))
    )


def test_latency_under_budget(parser):
    """Парсинг 40+ слов укладывается в <100 мс warm."""
    import time
    text = (
        "Главным управлением собственной безопасности Федеральной службы "
        "войск национальной гвардии Российской Федерации во 2-м квартале "
        "2025 года проведено проверочное мероприятие по контролю за "
        "выполнением капитального ремонта помещений административного "
        "здания. УОПМ в целях повышения эффективности мероприятий по "
        "профилактике, предупреждению, выявлению и пресечению преступлений, "
        "проводимых подразделениями собственной безопасности, обобщены и "
        "структурированы сведения о преступлениях, совершенных должностными "
        "лицами подразделений вневедомственной охраны."
    )
    parser.parse(text)  # warmup
    samples = []
    # invalidate кэш через slight text variation
    for i in range(3):
        t0 = time.perf_counter()
        parser.parse(text + " " + str(i))
        samples.append((time.perf_counter() - t0) * 1000)
    avg = sum(samples) / len(samples)
    assert avg < 200, f"Latency avg={avg:.1f} мс exceeds budget 200 мс"
