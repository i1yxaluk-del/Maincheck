from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'server/local'))
from decision_engine import DecisionEngine
from legal_style_stage import LegalStyleStage
from safe_diff import diff_candidates

def correct(text):return DecisionEngine().apply(text,LegalStyleStage().candidates(text))[0]
def main():
 duplicate='Проверкой\nустановлено, что в отношении Центра в\nотношении Центра не разработаны нормы\nснабжения хозяйственными товарами и\nинвентарем.'
 fixed=correct(duplicate);assert 'в отношении Центра\nне разработаны' in fixed;assert fixed.count('\n')==duplicate.count('\n')
 government='Проверкой\nустановлено, что в отношениях\nЦентра не разработаны нормы снабжения.';assert 'в отношении\nЦентра' in correct(government)
 assert correct('Организация состоит в отношениях с Центром.')=='Организация состоит в отношениях с Центром.'
 assert correct('В соответствие с приказом подготовлен отчет.')=='В соответствии с приказом подготовлен отчет.'
 source='вынесено постановление по делу об административном правонарушении по статье 12.6 Кодекса Российской Федерации об административных правонарушениях, на сумму 1 500 рублей.'
 expected=source.replace('правонарушениях, на сумму','правонарушениях на сумму');assert correct(source)==expected
 edits=diff_candidates(source,expected,'russian-gec-reasoning',.86);assert edits and 'bounded token diff' not in edits[0].reason;assert 'на сумму 1 500 рублей' in edits[0].reason
 print('V20 LEGAL STYLE AND EXPLANATIONS OK')
if __name__=='__main__':main()
