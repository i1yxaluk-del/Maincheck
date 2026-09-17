from __future__ import annotations
import argparse,json,re,statistics,sys,time,urllib.error,urllib.request,uuid
from pathlib import Path
from .protocol import parse_corrected
DEFAULT_CORPUS=Path(__file__).with_name('cases.jsonl');DELIMITERS=set('()[]{}«»„“”"')
def _delimiter_signature(text):return ''.join(ch for ch in text if ch in DELIMITERS)
def load_cases(path=None):
 requested=Path(path) if path else DEFAULT_CORPUS;target=requested if requested.exists() else DEFAULT_CORPUS
 if target!=requested:print(f'V20 CORPUS FALLBACK requested={requested} using={target}',file=sys.stderr)
 cases=[json.loads(x) for x in target.read_text(encoding='utf-8').splitlines() if x.strip()];ids=[c.get('id') for c in cases]
 if not cases or len(ids)!=len(set(ids)) or not all(ids):raise ValueError('corpus needs unique ids')
 if not any(c.get('clean') for c in cases) or not any(not c.get('clean') for c in cases):raise ValueError('corpus needs positive and clean controls')
 deep=any('expected_text' in c or 'error_count' in c for c in cases)
 if deep:
  for c in cases:
   if 'expected_text' not in c or 'error_count' not in c:raise ValueError(f"deep case {c.get('id')} needs expected_text and error_count")
   errors=int(c['error_count'])
   if c.get('clean') and errors!=0:raise ValueError(f"clean case {c['id']} must have zero errors")
   if not c.get('clean') and not 1<=errors<=3:raise ValueError(f"positive case {c['id']} must have 1-3 errors")
   if (c['text']==c['expected_text']) != bool(c.get('clean')):raise ValueError(f"clean/expected mismatch in {c['id']}")
   if c['text'].count('\n')!=c['expected_text'].count('\n'):raise ValueError(f"layout mismatch in expected text for {c['id']}")
   if c.get('preserve_delimiters',True) and _delimiter_signature(c['text'])!=_delimiter_signature(c['expected_text']):raise ValueError(f"delimiter mismatch in expected text for {c['id']}")
 return cases
def multipart(text,context=''):
 boundary='----v20'+uuid.uuid4().hex;chunks=[]
 for name,value in [('text',text),('context',context)]:chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.txt"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n'.encode(),value.encode(),b'\r\n']
 chunks.append(f'--{boundary}--\r\n'.encode());return b''.join(chunks),boundary
def _changes(payload):
 m=re.search(r'===CHANGES===\n(.*?)\n===END===',payload,re.S);return [x.strip() for x in (m.group(1).splitlines() if m else []) if x.strip()]
def _groups(rows,key):
 result={}
 for name in sorted({str(r.get(key) or 'unspecified') for r in rows}):
  part=[r for r in rows if str(r.get(key) or 'unspecified')==name];positive=[r for r in part if not r['clean']];clean=[r for r in part if r['clean']]
  result[name]={'cases':len(part),'passed':sum(r['pass'] for r in part),'pass_rate':round(sum(r['pass'] for r in part)/len(part),3),'positive_recall':round(sum(r['pass'] for r in positive)/len(positive),3) if positive else None,'clean_preservation':round(sum(r['pass'] for r in clean)/len(clean),3) if clean else None}
 return result
def run(url,cases):
 rows=[]
 for c in cases:
  body,b=multipart(c['text'],c.get('context',''));req=urllib.request.Request(url.rstrip('/')+'/suggest',body,{'Content-Type':f'multipart/form-data; boundary={b}'});t=time.perf_counter();payload='';corrected='';error=''
  try:payload=urllib.request.urlopen(req,timeout=c.get('timeout',90)).read().decode();corrected=parse_corrected(payload)
  except urllib.error.HTTPError as exc:error=f'HTTP {exc.code}: {exc.read().decode("utf-8","replace")[:2000]}'
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  ms=(time.perf_counter()-t)*1000;missing=[x for x in c.get('must_contain',[]) if x not in corrected];forbidden=[x for x in c.get('must_not_contain',[]) if x in corrected];response_missing=[x for x in c.get('must_response_contain',[]) if x not in payload];response_forbidden=[x for x in c.get('must_response_not_contain',[]) if x in payload]
  layout=corrected.count('\n')==c['text'].count('\n');expected=c.get('expected_text');exact=(corrected==expected) if expected is not None else ((corrected==c['text']) if c.get('clean') else True);delimiters=(not c.get('preserve_delimiters',True)) or _delimiter_signature(corrected)==_delimiter_signature(c['text']);reason='bounded token diff' not in payload.lower();changes=_changes(payload)
  passed=bool(not error and not missing and not forbidden and not response_missing and not response_forbidden and layout and exact and delimiters and reason)
  rows.append({'id':c['id'],'category':c.get('category','unspecified'),'source':c.get('source','unspecified'),'clean':bool(c.get('clean')),'expected_error_count':c.get('error_count'),'reported_change_count':0 if changes==['1. Ошибок не найдено.'] else len(changes),'pass':passed,'exact_ok':exact,'delimiter_ok':delimiters,'layout_ok':layout,'reason_ok':reason,'changed':corrected!=c['text'] if corrected else False,'latency_ms':round(ms,1),'missing_required':missing,'present_forbidden':forbidden,'response_missing':response_missing,'response_forbidden':response_forbidden,'changes':changes,'corrected':corrected,'expected_text':expected,'error':error})
 times=[r['latency_ms'] for r in rows];pos=[r for r in rows if not r['clean']];clean=[r for r in rows if r['clean']];passed=sum(r['pass'] for r in rows);p95=sorted(times)[max(0,int(.95*len(times))-1)]
 return {'cases':len(rows),'passed':passed,'pass_rate':round(passed/len(rows),3),'positive_recall':round(sum(r['pass'] for r in pos)/len(pos),3),'clean_preservation':round(sum(r['pass'] for r in clean)/len(clean),3),'false_positive_rate':round(sum(r['changed'] for r in clean)/len(clean),3),'exact_match_rate':round(sum(r['exact_ok'] for r in rows)/len(rows),3),'delimiter_preservation':round(sum(r['delimiter_ok'] for r in rows)/len(rows),3),'layout_preservation':round(sum(r['layout_ok'] for r in rows)/len(rows),3),'reason_quality':round(sum(r['reason_ok'] for r in rows)/len(rows),3),'internal_reason_leaks':sum(not r['reason_ok'] for r in rows),'errors':sum(bool(r['error']) for r in rows),'median_ms':round(statistics.median(times),1),'p95_ms':p95,'by_category':_groups(rows,'category'),'by_source':_groups(rows,'source'),'rows':rows}
def self_test():
 sample='===CORRECTED===\nтекст\n===CHANGES===\n1. Ошибок не найдено.\n===END===';assert parse_corrected(sample)=='текст';print('V20 BENCHMARK SELFTEST OK')
def main():
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--validate-corpus');p.add_argument('--url');p.add_argument('--corpus',default=str(DEFAULT_CORPUS));p.add_argument('--out');a=p.parse_args()
 if a.self_test:self_test();return 0
 cases=load_cases(a.validate_corpus or a.corpus)
 if a.validate_corpus:print(f'V20 CORPUS OK cases={len(cases)}');return 0
 if not a.url:p.error('--url is required')
 result=run(a.url,cases);blob=json.dumps(result,ensure_ascii=False,indent=2);print(blob)
 if a.out:Path(a.out).write_text(blob,encoding='utf-8')
 return 0
if __name__=='__main__':raise SystemExit(main())
