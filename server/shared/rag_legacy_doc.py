"""Безопасное чтение старых бинарных .doc через временную конвертацию LibreOffice."""
from __future__ import annotations
import shutil,subprocess,tempfile
from pathlib import Path


def read_legacy_doc(path: Path) -> str:
    """Конвертирует .doc во временный .docx, не изменяя исходный документ."""
    soffice=shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("Для загрузки .doc необходим LibreOffice (soffice); либо сохраните файл как .docx")
    with tempfile.TemporaryDirectory(prefix="maincheck-rag-doc-") as tmp:
        out=Path(tmp)/"out";profile=Path(tmp)/"profile";out.mkdir();profile.mkdir()
        profile_uri=profile.resolve().as_uri()
        command=[soffice,f"-env:UserInstallation={profile_uri}","--headless","--convert-to","docx","--outdir",str(out),str(path)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=180,check=False)
        converted=out/(path.stem+".docx")
        if result.returncode!=0 or not converted.exists():
            detail=(result.stderr or result.stdout or "файл DOCX не создан").strip()
            raise RuntimeError(f"LibreOffice не смог преобразовать .doc: {detail[:1000]}")
        from .garant_cleanup import _read_docx
        return _read_docx(converted)
