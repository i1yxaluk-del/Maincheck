from decision_engine import DecisionEngine,EditCandidate
from government_frame_stage import GovernmentFrameStage
from legal_style_stage import LegalStyleStage
from production_safety import ProductionSafety

def apply(stage,text):return DecisionEngine().apply(text,stage.candidates(text))[0]
def test_department():
 text='В\nсвоих объяснениях начальник отдела\nавтоматизация\nи обеспечения научной деятельности указал причины.';assert 'отдела\nавтоматизации\nи обеспечения' in apply(GovernmentFrameStage(),text)
def test_section():assert 'в разделе 2' in apply(GovernmentFrameStage(),'Сведения приведены в раздел 2 настоящего документа.')
def test_training():assert 'по огневой подготовке' in apply(GovernmentFrameStage(),'Занятия по огневой подготовок проведены своевременно.')
def test_subject():assert 'выявлены три нарушения' in apply(GovernmentFrameStage(),'По результатам проверки выявлена три нарушения.')
def test_blagodarya():assert 'благодаря принятым мерам' in apply(GovernmentFrameStage(),'Нарушение устранено благодаря принятых мер.')
def test_spelling():assert apply(LegalStyleStage(),'Вследствии проверки выявлены нарушения.')=='Вследствие проверки выявлены нарушения.'
def test_safety():
 safety=ProductionSafety();quoted='В разделе «Учебные предметы» указаны СВТ.';q=quoted.index('«');assert not safety.allowed(quoted,EditCandidate('«','"',.8,'sage-spell-punc',start=q))
 enumerated='1. Проверить оборудование;\n2. доложить о результатах.';semi=enumerated.index(';');assert not safety.allowed(enumerated,EditCandidate(';','.',.8,'sage-rupunct',start=semi))
def test_complex():
 text='Использование заказчиком Н(М)ЦК одного источника информации о ценах планируемых к приобретению товаров (оказанию услуг) приводит к завышению Н(М)ЦК.'
 def c(before,after):return EditCandidate(before,after,.8,'sage-rupunct',start=text.index(before),sources=('sage-rupunct','sage-spell-punc'))
 candidates=[c('ценах','ценах,'),c('товаров (','товаров, ('),c('услуг) приводит','услуг), приводит'),c('завышению Н','завышению (Н')]
 assert ProductionSafety().filter(text,candidates)==[]
def test_clean():
 stage=GovernmentFrameStage()
 for text in ('Организация состоит в отношениях с Центром.','Начальник отдела автоматизации и обеспечения научной деятельности представил объяснения.','Документ подготовлен согласно приказу директора.'):assert apply(stage,text)==text
