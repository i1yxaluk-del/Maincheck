from contextual_agreement import ContextualAgreementRules

def test_copular_predicative_agreement_across_context():
 text='что входная дверь в Центр в ночное время\nоставалась открытыми\nнастежь'
 found=ContextualAgreementRules().candidates(text)
 assert any(c.before=='открытыми' and c.after=='открытой' for c in found)
