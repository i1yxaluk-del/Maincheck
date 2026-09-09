from __future__ import annotations

import asyncio
import sys
import time

from hybrid_editor import HybridRouter

CASES = [
    ("spelling", "Сегодня сотрудник выполнил упражнение в соотвествии с планом."),
    ("punctuation", "При проверке документа установлено что сведения представлены не полностью."),
    ("agreement", "План профессиональной служебная и физической подготовки утвержден."),
    ("government", "Для выполнения нескольких упражнения требуется подготовка."),
    ("quantifier", "При выполнении сотрудником нескольких вида оружия допускается..."),
    ("case", "Доклад направлен в соответствии с указанием руководителя."),
    ("compound", "Сотрудник прибыл доложил о готовности к занятию."),
    ("verb", "Сотрудники обязаны обеспечить выполнение требований и контролирует результат."),
]


async def main() -> None:
    preset = (sys.argv[1] if len(sys.argv) > 1 else "X").upper()
    router = HybridRouter(preset)
    started = time.perf_counter()
    await router.warmup()
    print(f"warmup_ms={int((time.perf_counter() - started) * 1000)}")
    for name, text in CASES:
        started = time.perf_counter()
        candidates = await router.candidates(text, "")
        elapsed = int((time.perf_counter() - started) * 1000)
        print(f"[{name}] {elapsed}ms candidates={len(candidates)}")
        for candidate in candidates[:8]:
            print(f"  {candidate.category}: {candidate.before!r} -> {candidate.after!r}")
    print("metrics:", router.metrics())


if __name__ == "__main__":
    asyncio.run(main())
