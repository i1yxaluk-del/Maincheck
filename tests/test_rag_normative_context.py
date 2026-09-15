from shared.rag_normative_context import plan_queries,render_evidence

def test_query_plan_uses_name_without_wrong_abbreviation():
 q=plan_queries('в распоряжении ФГКУ «НИЦ «Охрана» Росгвардии')
 assert 'ФГКУ' in q
 assert any('НИЦ' in x and 'Охрана' in x for x in q)

def test_evidence_requires_applicability_in_prompt():
 text=render_evidence([{'doc_id':'order','chunk_id':2,'excerpt':'В случае X применяется Y.','normative':True}])
 assert 'область применения' in text
 assert 'В случае X' in text
