from pathlib import Path
import pytest
from shared.rag_office_extract import _validate

def test_mojibake_is_rejected():
    with pytest.raises(RuntimeError,match="U\\+FFFD"):_validate("����"*20,Path("bad.rtf"))

def test_normal_russian_is_accepted():
    text="Приказ устанавливает порядок ведения учета."
    assert _validate(text,Path("ok.rtf"))==text
