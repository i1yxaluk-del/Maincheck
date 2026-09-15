from logical_segmentation import split_logical_sentences
from shared.rag_context import should_use_rag

def test_visual_wraps_are_not_sentences():
 text='В\nходе проверки обнаружен тот факт\nчто дверь оставалась открытой. Следующее предложение.'
 parts=split_logical_sentences(text)
 assert len(parts)==2
 assert 'дверь оставалась' in parts[0].text

def test_paragraph_break_is_boundary():
 assert len(split_logical_sentences('Первый абзац\n\nВторой абзац'))==2

def test_generic_grammar_does_not_trigger_rag():
 text='В ходе просмотра видеозаписей дверь оставалась открытой настежь.'
 assert not should_use_rag(text)
 assert should_use_rag('Наименование ФГКУ «НИЦ «Охрана» Росгвардии')
