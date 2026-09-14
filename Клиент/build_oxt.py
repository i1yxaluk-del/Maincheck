"""Build a safer LibreOffice extension without committing binary archives.

The source tree remains readable XML/Basic. During packaging we apply three
small compatibility fixes to the legacy Main.xba and produce a deterministic
OXT that CI publishes as an artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "AI_Suggester"
VERSION = "1.9.0"


def patched_main(text: str) -> str:
    replacements = [
        (
            'If aOldFrags(k) <> "" And aNewFrags(k) <> "" Then nFrags = nFrags + 1',
            'If aOldFrags(k) <> "" Then nFrags = nFrags + 1',
        ),
        (
            "    Else\n        ' Нет структурированных правок — применяем целиком, но аккуратно\n"
            "        ApplyWholeReplace oSel, sCorrected\n    End If",
            "    Else\n        ' Fail closed: never rewrite a formatted selection without local edits.\n"
            "        MsgBox \"Сервер не вернул безопасных локальных правок. Текст не изменен.\", 48, \"AI Suggester\"\n"
            "    End If",
        ),
        (
            "        Dim nPos As Long : nPos = InStr(sCurText, sOld)\n"
            "        If nPos = 0 Then",
            "        Dim nPos As Long : nPos = InStr(sCurText, sOld)\n"
            "        Dim nNextPos As Long : nNextPos = 0\n"
            "        If nPos > 0 Then nNextPos = InStr(nPos + Len(sOld), sCurText, sOld)\n"
            "        If nNextPos > 0 Then\n"
            "            sSkipped = sSkipped & \"  • неоднозначный фрагмент: «\" & Left(sOld, 70) & \"»\" & Chr(10)\n"
            "            GoTo NextFrag\n"
            "        End If\n"
            "        If nPos = 0 Then",
        ),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected exactly one Main.xba patch target: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def patched_description(text: str) -> str:
    marker = '<version value="1.8.3"/>'
    if text.count(marker) != 1:
        raise RuntimeError("Unexpected extension source version")
    return text.replace(marker, f'<version value="{VERSION}"/>')


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(SOURCE).as_posix()
            data = path.read_bytes()
            if relative == "ai_macro/Main.xba":
                data = patched_main(data.decode("utf-8")).encode("utf-8")
            elif relative == "description.xml":
                data = patched_description(data.decode("utf-8")).encode("utf-8")
            info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / f"AI_Suggester-{VERSION}.oxt")
    args = parser.parse_args()
    build(args.output)
    print(args.output)
