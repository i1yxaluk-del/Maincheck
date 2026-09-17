from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'server/local'))
from decision_engine import DecisionEngine
from legal_style_stage import LegalStyleStage

def correct(text):return DecisionEngine().apply(text,LegalStyleStage().candidates(text))[0]

def main():
 duplicate='Проверкой\nустановлено, что в отношении Центра в\nотношении Центра не разработаны нормы\nснабжения хозяйственными товарами и\nинвентарем.'
 fixed=correct(duplicate);assert 'в отношении Центра\nне разработаны' in fixed;assert fixed.count('\n')==duplicate.count('\n')
 government='Проверкой\nустановлено, что в отношениях\nЦентра не разработаны нормы снабжения.'
 assert 'в отношении\nЦентра' in correct(government)
 assert correct('Организация состоит в отношениях с Центром.')=='Организация состоит в отношениях с Центром.'
 assert correct('В соответствие с приказом подготовлен отчет.')=='В соответствии с приказом подготовлен отчет.'
 print('V20 LEGAL STYLE OK')
if __name__=='__main__':main()
