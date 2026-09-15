"""Надёжное извлечение старых DOC/RTF через фильтры LibreOffice."""
from __future__ import annotations
import os,shutil,subprocess,tempfile
from pathlib import Path


def _soffice() -> str:
    configured=os.getenv("RAG_SOFFICE_PATH","").strip()
    candidates=[configured,shutil.which("soffice") or "",shutil.which("libreoffice") or "",
                "/usr/bin/soffice","/usr/lib/libreoffice/program/soffice","/opt/libreoffice/program/soffice"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():return candidate
    raise RuntimeError("Для DOC/RTF необходим LibreOffice (soffice) или RAG_SOFFICE_PATH")


def _validate(text: str, source: Path) -> str:
    if not text.strip():raise RuntimeError(f"После преобразования документ пуст: {source.name}")
    bad=text.count("\ufffd")
    if bad and bad/max(1,len(text))>0.0001:
        raise RuntimeError(f"Извлечение {source.name} содержит {bad} символов замены U+FFFD; индексация отменена")
    return text


def read_office_legacy(path: Path) -> str:
    """Преобразует DOC/RTF во временный DOCX с корректной RTF codepage."""
    with tempfile.TemporaryDirectory(prefix="maincheck-rag-office-") as tmp:
        root=Path(tmp);out=root/"out";profile=root/"profile";out.mkdir();profile.mkdir()
        command=[_soffice(),f"-env:UserInstallation={profile.resolve().as_uri()}","--headless","--convert-to","docx","--outdir",str(out),str(path)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=300,check=False)
        converted=out/(path.stem+".docx")
        if result.returncode or not converted.exists():
            detail=(result.stderr or result.stdout or "DOCX не создан").strip()
            raise RuntimeError(f"LibreOffice не преобразовал {path.name}: {detail[:1000]}")
        from .garant_cleanup import _read_docx
        return _validate(_read_docx(converted),path)
