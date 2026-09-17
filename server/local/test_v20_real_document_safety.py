from decision_engine import EditCandidate
from production_safety import ProductionSafety


def candidate(text,before,after,category):
    return EditCandidate(before,after,0.99,category,'test',start=text.index(before))


def test_rejects_generated_word_damage_and_taints_family():
    text='как с единственным поставщиком, Центром заключены два государственных контракта'
    damaged=candidate(text,'поставщиком, ','ппоставщиком','sage-spell-punc')
    sibling=candidate(text,'заключены','заключены,','sage-rupunct')
    safety=ProductionSafety()
    assert safety.filter(text,[damaged,sibling]) == []
    metrics=safety.metrics()
    assert metrics['word_boundary'] == 1
    assert metrics['tainted_sentence'] == 1


def test_preserves_correct_numeral_government():
    text='Центром заключены два государственных контракта.'
    wrong=candidate(text,'государственных','государственного','rule-agreement')
    safety=ProductionSafety()
    assert safety.filter(text,[wrong]) == []
    assert safety.metrics()['numeral_agreement'] == 1


def test_keeps_proven_deterministic_government():
    text='Согласно приказа директора комиссия продолжила работу.'
    fixed=candidate(text,'приказа','приказу','rule-dative-frame')
    assert ProductionSafety().filter(text,[fixed]) == [fixed]
