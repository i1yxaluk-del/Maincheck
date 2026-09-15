from __future__ import annotations
import asyncio,os,re
from .api import make_app
from .protocol import Edit,apply_edits

class OpenVinoEngine:
    name='openvino-seq2seq'
    def __init__(self): self.model_id=os.getenv('V20_OPENVINO_MODEL','ai-forever/sage-fredt5-large'); self.model=None; self.tokenizer=None; self.error='not loaded'; self.attempted=False
    def _load(self):
        if self.model is not None:return
        if self.attempted:raise RuntimeError(self.error)
        self.attempted=True
        try:
            from transformers import AutoTokenizer
            from optimum.intel.openvino import OVModelForSeq2SeqLM
            self.tokenizer=AutoTokenizer.from_pretrained(self.model_id)
            self.model=OVModelForSeq2SeqLM.from_pretrained(self.model_id,export=True,compile=True)
            self.error=''
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}';raise
    def _infer(self,text):
        self._load(); flat=text.replace('\n',' '); inputs=self.tokenizer(flat,return_tensors='pt',truncation=True,max_length=512)
        out=self.model.generate(**inputs,max_new_tokens=min(256,max(32,len(inputs['input_ids'][0])+32)),num_beams=1)
        corrected=self.tokenizer.decode(out[0],skip_special_tokens=True).strip()
        from difflib import SequenceMatcher
        src=list(re.finditer(r'[А-Яа-яЁёA-Za-z0-9-]+',text)); dst=list(re.finditer(r'[А-Яа-яЁёA-Za-z0-9-]+',corrected)); edits=[]
        for tag,i1,i2,j1,j2 in SequenceMatcher(None,[x.group() for x in src],[x.group() for x in dst],autojunk=False).get_opcodes():
            if tag=='replace' and i2-i1==j2-j1==1:
                before=src[i1].group(); edits.append(Edit(src[i1].start(),src[i1].end(),before,dst[j1].group(),'OpenVINO seq2seq',.80))
        return apply_edits(text,edits)
    async def correct(self,text,context): return await asyncio.to_thread(self._infer,text)
    async def health(self):
        try: await asyncio.to_thread(self._load); return {'ready':True,'engine':self.name,'model':self.model_id}
        except Exception:return {'ready':False,'engine':self.name,'model':self.model_id,'error':self.error}
app=make_app(OpenVinoEngine())
