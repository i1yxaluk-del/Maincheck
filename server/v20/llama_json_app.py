from __future__ import annotations
import json,os,re
import httpx
from .api import make_app
from .protocol import Edit,apply_edits

class LlamaJsonEngine:
    name='llama.cpp-json-edits'
    def __init__(self):
        self.url=os.getenv('V20_LLAMA_URL','http://127.0.0.1:8091').rstrip('/')
        self.model=os.getenv('V20_LLAMA_MODEL_NAME','local-gec')
        self.timeout=float(os.getenv('V20_LLAMA_TIMEOUT','25'))
        self.smoke_ok=False; self.smoke_error='not run'
    async def _json(self,text,context,max_tokens=180):
        system='Ты корректор русского официального текста. Верни только JSON-объект {"edits":[{"start":0,"end":1,"before":"...","after":"...","reason":"...","confidence":0.9}]}. Смещения символов относятся к исходному тексту. Не переписывай корректные фрагменты, названия, числа и переносы. Если правок нет, верни {"edits":[]}.'
        user=f'Контекст:\n{context[-1200:]}\n\nТекст:\n{text}\n/no_think'
        payload={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'temperature':0,'max_tokens':max_tokens,'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':False}}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r=await c.post(self.url+'/v1/chat/completions',json=payload)
            if r.is_error: raise RuntimeError(f'llama.cpp HTTP {r.status_code}: {r.text[:1000]}')
            raw=r.json()
        choice=raw.get('choices',[{}])[0]; message=choice.get('message') or {}; content=message.get('content') or ''; reasoning=message.get('reasoning_content') or ''
        if not content.strip(): raise RuntimeError(f'empty JSON content finish={choice.get("finish_reason")} reasoning_chars={len(reasoning)} reasoning_preview={reasoning[:300]!r}')
        try: return json.loads(content)
        except json.JSONDecodeError as first:
            m=re.search(r'\{.*\}',content,re.S)
            if not m: raise RuntimeError(f'non-JSON llama.cpp content: {content[:500]!r}') from first
            try:return json.loads(m.group(0))
            except json.JSONDecodeError as second:raise RuntimeError(f'invalid JSON content: {content[:500]!r}') from second
    async def health(self):
        try:
            async with httpx.AsyncClient(timeout=3) as c:r=await c.get(self.url+'/health');r.raise_for_status()
            if not self.smoke_ok:
                data=await self._json('Текст без ошибок.','',max_tokens=64)
                if not isinstance(data,dict) or not isinstance(data.get('edits'),list):raise RuntimeError(f'invalid smoke JSON: {data!r}')
                self.smoke_ok=True; self.smoke_error=''
            return {'ready':True,'engine':self.name,'backend':self.url,'smoke':'ok'}
        except Exception as exc:self.smoke_error=f'{type(exc).__name__}: {exc}';return {'ready':False,'engine':self.name,'backend':self.url,'smoke':'failed','error':self.smoke_error}
    async def correct(self,text,context):
        data=await self._json(text,context); edits=[]
        for x in data.get('edits',[]) if isinstance(data,dict) else []:
            try: edits.append(Edit(int(x['start']),int(x['end']),str(x['before']),str(x['after']),str(x.get('reason','LLM')),float(x.get('confidence',.75))))
            except Exception: continue
        return apply_edits(text,edits)
app=make_app(LlamaJsonEngine())
