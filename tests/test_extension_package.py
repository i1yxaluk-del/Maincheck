"""Валидация структуры расширения LibreOffice (.oxt)."""
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "Клиент"
EXT_DIR = CLIENT / "AI_Suggester"
OXT = CLIENT / "AI_Suggester.oxt"


def _parse(p: Path) -> ET.Element:
    return ET.parse(p).getroot()


def test_description_version():
    root = _parse(EXT_DIR / "description.xml")
    version = root.find("{http://openoffice.org/extensions/description/2006}version")
    assert version is not None
    # v1.8.3: диалог управления словарём (ListBox + Кнопки
    # Удалить/Добавить/Закрыть) + новая тулбар-кнопка
    # m003 «AI: Словарь».
    assert version.get("value") == "1.8.3"


def test_manifest_lists_library_and_xcu():
    root = _parse(EXT_DIR / "META-INF" / "manifest.xml")
    ns = "{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}"
    paths = {e.get(f"{ns}full-path") for e in root.findall(f"{ns}file-entry")}
    assert "ai_macro/" in paths
    assert "Addons.xcu" in paths


def test_script_xlb_lists_modules():
    root = _parse(EXT_DIR / "ai_macro" / "script.xlb")
    ns = "{http://openoffice.org/2000/library}"
    names = {e.get("library:name") or e.get(f"{ns}name") for e in root.findall(f"{ns}element")}
    assert {"Main", "Settings", "Health", "Dict"}.issubset(names)


@pytest.mark.parametrize("name", ["Main.xba", "Settings.xba", "Health.xba", "Dict.xba"])
def test_basic_modules_are_parseable(name):
    p = EXT_DIR / "ai_macro" / name
    assert p.exists(), f"Отсутствует {p}"
    # XML корректный (несмотря на CDATA)
    ET.parse(p)
    body = p.read_text(encoding="utf-8")
    # Обёртка CDATA присутствует (защита от & в Basic-коде)
    assert "<![CDATA[" in body
    assert "]]>" in body


def test_addons_xcu_has_user_toolbar_entries():
    """
    На панели сотрудника три кнопки (v1.8.3):
      m001 — Main.AISuggestSelection («AI: Улучшить текст»).
      m002 — Dict.AIDictAddSelection («AI: В словарь», v1.8b) —
             быстрое добавление выделенного фрагмента.
      m003 — Dict.AIDictManage («AI: Словарь», v1.8.3) — диалог
             с ListBox и кнопками Удалить/Добавить/Закрыть.
    Диагностический Health.AICheckServer и legacy Dict.AIDictListWords /
    Dict.AIDictRemoveWord на панель не вынесены — по прежнему
    доступны через меню макросов.
    """
    root = _parse(EXT_DIR / "Addons.xcu")
    ns = "{http://openoffice.org/2001/registry}"
    nodes = list(root.iter("node"))
    names = {n.get(f"{ns}name") for n in nodes}
    assert "m001" in names  # AISuggestSelection
    assert "m002" in names  # AIDictAddSelection (v1.8b)
    assert "m003" in names  # AIDictManage (v1.8.3)

    # Проверяем что каждый нод ссылается на правильный макрос
    expected_urls = {
        "m001": "Main.AISuggestSelection",
        "m002": "Dict.AIDictAddSelection",
        "m003": "Dict.AIDictManage",
    }
    found = {}
    for node in nodes:
        name = node.get(f"{ns}name")
        if name in expected_urls:
            urls = [v.text or "" for v in node.iter() if v.tag.endswith("value")]
            assert any(expected_urls[name] in u for u in urls), (
                f"{name} должна вызывать {expected_urls[name]}, найдены: {urls}"
            )
            found[name] = True
    assert set(found) == set(expected_urls), (
        f"Не все кнопки найдены: проверено {found}, ожидалось {set(expected_urls)}"
    )


def test_oxt_artifact_can_be_rebuilt(tmp_path):
    """Собираем .oxt из Клиент/AI_Suggester/ и проверяем, что архив валидный."""
    out = tmp_path / "AI_Suggester.oxt"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in EXT_DIR.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(EXT_DIR).as_posix())
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        for expected in (
            "description.xml",
            "Addons.xcu",
            "META-INF/manifest.xml",
            "ai_macro/Main.xba",
            "ai_macro/Settings.xba",
            "ai_macro/Health.xba",
            "ai_macro/Dict.xba",
            "ai_macro/script.xlb",
            "ai_macro/dialog.xlb",
        ):
            assert expected in names, f"В .oxt нет {expected}"


def test_dict_xba_has_manage_dialog_v183():
    """v1.8.3: новый Sub AIDictManage() и helper'ы для диалога управления
    словарём (ListBox + Кнопки). Listener для STANDARD-кнопки «Добавить»
    реализован через CreateUnoListener с префиксом AIDictDlg_BtnAdd_,
    значит обязательно должны быть Sub'ы actionPerformed/disposing с этим
    префиксом — иначе диалог упадёт на клике."""
    body = (EXT_DIR / "ai_macro" / "Dict.xba").read_text(encoding="utf-8")
    # Точка входа на тулбаре
    assert "Sub AIDictManage()" in body
    # Builder / dispatcher
    assert "AIDictManage_ShowDialog" in body
    assert "AIDictManage_DoAdd" in body
    assert "AIDictManage_DoDelete" in body
    # Loader
    assert "LoadDictWordsJoined" in body
    # Listener для кнопки «Добавить новое...» — оба метода XActionListener
    # (actionPerformed + disposing inherited from XEventListener) должны быть.
    # createUnoListener будет искать Sub-ы по этому префиксу.
    assert "AIDictDlg_BtnAdd_actionPerformed" in body
    assert "AIDictDlg_BtnAdd_disposing" in body
    # Используется правильный UNO-сервис
    assert "com.sun.star.awt.UnoControlDialogModel" in body
    assert "com.sun.star.awt.UnoControlListBoxModel" in body
    # PushButtonType: OK=1 (Delete), CANCEL=2 (Close), STANDARD=0 (Add)
    assert "PushButtonType = 1" in body
    assert "PushButtonType = 2" in body
    # MultiSelection для ListBox
    assert "MultiSelection = True" in body


def test_main_xba_uses_settings_module():
    body = (EXT_DIR / "ai_macro" / "Main.xba").read_text(encoding="utf-8")
    assert "Settings.GetServerList()" in body
    assert "Settings.GetUseTrackChanges()" in body
    assert "ApplyCorrection" in body
    assert "RecordChanges" in body
    # HTTP-status-code проверка
    assert "-w" in body and "http_code" in body


def test_committed_oxt_is_installable():
    """Собранный артефакт в корне должен быть валидным zip-архивом LibreOffice."""
    if not OXT.exists():
        pytest.skip("Клиент/AI_Suggester.oxt ещё не пересобран — пропускаем")
    with zipfile.ZipFile(OXT) as z:
        names = z.namelist()
    assert "description.xml" in names
    assert "META-INF/manifest.xml" in names
