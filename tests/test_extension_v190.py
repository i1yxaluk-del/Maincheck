import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "Клиент" / "build_oxt.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_oxt", BUILD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v190_package_fails_closed_and_supports_deletions(tmp_path):
    builder = load_builder()
    output = tmp_path / "AI_Suggester-1.9.0.oxt"
    builder.build(output)
    with zipfile.ZipFile(output) as archive:
        main = archive.read("ai_macro/Main.xba").decode("utf-8")
        description = archive.read("description.xml").decode("utf-8")
    assert '<version value="1.9.0"/>' in description
    assert 'If aOldFrags(k) <> "" Then nFrags = nFrags + 1' in main
    assert 'And aNewFrags(k) <> ""' not in main
    assert "ApplyWholeReplace oSel, sCorrected" not in main
    assert "не вернул безопасных локальных правок" in main
    assert "nNextPos" in main and "неоднозначный фрагмент" in main


def test_v190_build_is_deterministic(tmp_path):
    builder = load_builder()
    first = tmp_path / "one.oxt"
    second = tmp_path / "two.oxt"
    builder.build(first)
    builder.build(second)
    assert first.read_bytes() == second.read_bytes()
