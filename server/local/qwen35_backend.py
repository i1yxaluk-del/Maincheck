from __future__ import annotations

from russian_gec_backend import RussianGecSpecialist


class Qwen35Backend:
    """Compatibility wrapper for the v5 measured Russian GEC specialist."""

    _shared_instance: "Qwen35Backend | None" = None

    def __new__(cls):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
            cls._shared_instance._backend = RussianGecSpecialist()
        return cls._shared_instance

    async def correct(self, text: str, context: str = "", temperature: float = 0.0) -> str:
        return await self._backend.correct(text)

    async def warmup(self) -> None:
        await self._backend.warmup()
