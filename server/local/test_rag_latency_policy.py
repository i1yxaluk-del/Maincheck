from shared.rag_normative_context import plan_queries

def test_query_plan_is_bounded():
 assert len(plan_queries('ФГКУ «НИЦ «Охрана» Росгвардии '*20))<=5

def test_pipeline_timeout_default_is_interactive(monkeypatch):
 monkeypatch.delenv('REQUEST_PIPELINE_TIMEOUT',raising=False)
 import decision_app_v12
 assert decision_app_v12._metrics()['request_pipeline_timeout']==30.0
