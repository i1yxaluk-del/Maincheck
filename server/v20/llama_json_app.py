from __future__ import annotations
import json,os,re
import httpx
from .api import make_app
from .protocol import Edit,apply_edits

class LlamaJsonEngine:
    name='llama.cpp-json-edits'
    def __init__(self): self.url=os.getenv('V20_LLAMA_URL','http://127.0.0.1:8091').rstrip('/'); self.model=os.getenv('V20_LLAMA_MODEL_NAME','local-gec'); self.timeout=float(os.getenv('V20_LLAMA_TIMEOUT','25'))
    async def health(self):
        try:
            async with httpx.AsyncClient(timeout=3) as c:r=await c.get(self.url+'/health');r.raise_for_status()
            return {'ready':True,'engine':self.name,'backend':self.url}
        except Exception as exc:return {'ready':False,'engine':self.name,'backend':self.url,'error':str(exc)}
    async def correct(self,text,context):
        system='Ты корректор русского официального текста. Верни только JSON-объект {"edits":[{"start":0,"end":1,"before":"...","after":"...","reason":"...","confidence":0.9}]}. Смещения символов относятся к исходному тексту. Не переписывай корректные фрагменты, названия, числа и переносы.'
        user=f'Контекст:\n{context[-1200:]}\n\nТекст:\n{text}'
        payload={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'temperature':0,'max_tokens':180,'response_format':{'type':'json_object'}}
        async with httpx.AsyncClient(timeout=self.timeout) as c:r=await c.post(self.url+'/v1/chat/completions',json=payload);r.raise_for_status();content=r.json()['choices'][0]['message']['content']
        m=re.search(r'\{.*\}',content,re.S); data=json.loads(m.group(0) if m else content); edits=[]
        for x in data.get('edits',[]) if isinstance(data,dict) else []:
            try: edits.append(Edit(int(x['start']),int(x['end']),str(x['before']),str(x['after']),str(x.get('reason','LLM')),float(x.get('confidence',.75))))
            except Exception: continue
        return apply_edits(text,edits)
app=make_app(LlamaJsonEngine())
