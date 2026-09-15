from contextual_safety import protect_contextual_commas
from decision_engine import EditCandidate

def test_model_cannot_remove_comma_before_uchityvaya_chto():
 text='дверь оставалась открытой настежь, учитывая, что сотрудник спал'
 c=EditCandidate('настежь, учитывая','настежь учитывая',.9,'sage-spell-punc','x',start=text.index('настежь'))
 assert protect_contextual_commas([c],text)==[]
