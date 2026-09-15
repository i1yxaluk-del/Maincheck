from __future__ import annotations
import argparse,json,statistics,sys,time,urllib.request,uuid
from pathlib import Path
from .protocol import parse_corrected

DEFAULT_CORPUS=Path(__file__).with_name('cases.jsonl')

def load_cases(path=None):
    requested=Path(path) if path else DEFAULT_CORPUS
    target=requested if requested.exists() else DEFAULT_CORPUS
    if target != requested: print(f'V20 CORPUS FALLBACK requested={requested} using={target}',file=sys.stderr)
    cases=[json.loads(x) for x in target.read_text(encoding='utf-8').splitlines() if x.strip()]
    ids=[c.get('id') for c in cases]
    if not cases or len(ids)!=len(set(ids)) or not all(ids):raise ValueError('corpus needs unique ids')
    if not any(c.get('clean') for c in cases) or not any(not c.get('clean') for c in cases):raise ValueError('corpus needs positive and clean controls')
    return cases

def multipart(text,context=''):
    boundary='----v20'+uuid.uuid4().hex; chunks=[]
    for name,value in [('text',text),('context',context)]:chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.txt"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n'.encode(),value.encode(),b'\r\n']
    chunks.append(f'--{boundary}--\r\n'.encode());return b''.join(chunks),boundary

def run(url,cases):
    rows=[]
    for c in cases:
        body,b=multipart(c['text'],c.get('context',''));req=urllib.request.Request(url.rstrip('/')+'/suggest',body,{'Content-Type':f'multipart/form-data; boundary={b}'})
        t=time.perf_counter()
        try: payload=urllib.request.urlopen(req,timeout=c.get('timeout',90)).read().decode(); corrected=parse_corrected(payload);error=''
        except Exception as exc: corrected='';error=str(exc)
        ms=(time.perf_counter()-t)*1000; missing=[x for x in c.get('must_contain',[]) if x not in corrected]; forbidden=[x for x in c.get('must_not_contain',[]) if x in corrected]; layout=corrected.count('\n')==c['text'].count('\n'); clean_ok=(corrected==c['text']) if c.get('clean') else True
        rows.append({'id':c['id'],'clean':bool(c.get('clean')),'pass':bool(not error and not missing and not forbidden and layout and clean_ok),'layout_ok':layout,'changed':corrected!=c['text'] if corrected else False,'latency_ms':round(ms,1),'missing_required':missing,'present_forbidden':forbidden,'corrected':corrected,'error':error})
    times=[r['latency_ms'] for r in rows]; pos=[r for r in rows if not r['clean']]; clean=[r for r in rows if r['clean']]; passed=sum(r['pass'] for r in rows);p95=sorted(times)[max(0,int(.95*len(times))-1)]
    return {'cases':len(rows),'passed':passed,'pass_rate':round(passed/len(rows),3),'positive_recall':round(sum(r['pass'] for r in pos)/len(pos),3),'clean_preservation':round(sum(r['pass'] for r in clean)/len(clean),3),'false_positive_rate':round(sum(r['changed'] for r in clean)/len(clean),3),'layout_preservation':round(sum(r['layout_ok'] for r in rows)/len(rows),3),'errors':sum(bool(r['error']) for r in rows),'median_ms':round(statistics.median(times),1),'p95_ms':p95,'rows':rows}
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
