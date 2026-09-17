from decision_engine import DecisionEngine,EditCandidate
from government_frame_stage import GovernmentFrameStage
from legal_style_stage import LegalStyleStage
from production_safety import ProductionSafety

def apply(stage,text):return DecisionEngine().apply(text,stage.candidates(text))[0]
def test_measured_grammar_failures():
 stage=GovernmentFrameStage()
 cases={'В\nсвоих объяснениях начальник отдела\nавтоматизация\nи обеспечения научной деятельности указал причины.':'отдела\nавтоматизации\nи обеспечения','Сведения приведены в раздел 2 настоящего документа.':'в разделе 2','Занятия по огневой подготовок проведены своевременно.':'по огневой подготовке','По результатам проверки выявлена три нарушения.':'выявлены три нарушения','Нарушение устранено благодаря принятых мер.':'благодаря принятым мерам'}
 for source,expected in cases.items():assert expected in apply(stage,source),(source,stage.candidates(source))
def test_spelling_guarded_rule():assert apply(LegalStyleStage(),'Вследствии проверки выявлены нарушения.')=='Вследствие проверки выявлены нарушения.'
def test_production_safety_blocks_quote_and_enumeration_formatting():
 safety=ProductionSafety();quoted='В разделе «Учебные предметы» указаны СВТ.';q=quoted.index('«');assert not safety.allowed(quoted,EditCandidate('«','"',.8,'sage-spell-punc',start=q))
 enumerated='1. Проверить оборудование;\n2. доложить о результатах.';semi=enumerated.index(';');assert not safety.allowed(enumerated,EditCandidate(';','.',.8,'sage-rupunct',start=semi))
def test_clean_frames_remain_clean():
 stage=GovernmentFrameStage()
 for text in ('Организация состоит в отношениях с Центром.','Начальник отдела автоматизации и обеспечения научной деятельности представил объяснения.','Документ подготовлен согласно приказу директора.'):
  assert apply(stage,text)==text
