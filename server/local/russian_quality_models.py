"""SAGE-корректор орфографии и пунктуации (v9)."""
from __future__ import annotations
import asyncio,logging,os
from dataclasses import dataclass
logger=logging.getLogger('ai_suggester.quality_models');DEFAULT_SPELL_MODEL='ai-forever/sage-fredt5-distilled-95m'
@dataclass(frozen=True)
class QualityModelStats:
 enabled:bool;loaded:bool;model:str;calls:int;quantized:bool=False
class SageRussianCorrector:
 def __init__(self):
  self.enabled=os.getenv('SAGE_CORRECTOR_ENABLED','true').lower() in {'1','true','yes','on'};self.model_id=os.getenv('SAGE_CORRECTOR_MODEL',DEFAULT_SPELL_MODEL);self.max_new_tokens=int(os.getenv('SAGE_CORRECTOR_MAX_NEW_TOKENS','192'));self.max_input_tokens=int(os.getenv('SAGE_CORRECTOR_MAX_INPUT_TOKENS','384'));self.num_beams=int(os.getenv('SAGE_CORRECTOR_NUM_BEAMS','1'));self.threads=int(os.getenv('SAGE_CORRECTOR_THREADS','8'));self.quantize=os.getenv('SAGE_QUANTIZE','none').strip().lower();self.max_batch=int(os.getenv('SAGE_CORRECTOR_MAX_BATCH','8'));self._tokenizer=None;self._model=None;self._generation_config=None;self._quantized=False;self._calls=0;self._lock=asyncio.Lock()
 @property
 def available(self):return self.enabled
 def _load(self):
  if self._model is not None:return
  import torch
  from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,GenerationConfig
  try:torch.set_num_threads(max(1,self.threads));torch.set_num_interop_threads(1)
  except RuntimeError:pass
  logger.info('Loading Russian spelling/punctuation specialist: %s',self.model_id);self._tokenizer=AutoTokenizer.from_pretrained(self.model_id);model=AutoModelForSeq2SeqLM.from_pretrained(self.model_id,dtype=torch.float32);model.eval()
  if self.quantize=='int8':
   try:model=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8);self._quantized=True;logger.info('SAGE: включена динамическая int8-квантизация')
   except Exception as exc:logger.warning('SAGE: int8-квантизация недоступна (%s), работаем в fp32',exc)
  self._model=model
  # Some model repositories ship max_length=256 in generation_config.json.
  # Explicitly null it: max_new_tokens is the only output-length limit.
  self._generation_config=GenerationConfig(max_length=None,max_new_tokens=self.max_new_tokens,num_beams=max(1,self.num_beams),do_sample=False,no_repeat_ngram_size=3,early_stopping=self.num_beams>1)
 async def correct(self,text):
  result=await self.correct_batch([text]);return result[0] if result else text
 async def correct_batch(self,texts):
  if not self.enabled:return list(texts)
  payload=[t for t in texts if t and t.strip()]
  if not payload:return list(texts)
  self._calls+=1
  async with self._lock:corrected=await asyncio.to_thread(self._correct_sync,payload)
  iterator=iter(corrected);return [next(iterator) if t and t.strip() else t for t in texts]
 def _correct_sync(self,texts):
  import torch
  self._load();out=[]
  for start in range(0,len(texts),max(1,self.max_batch)):
   chunk=[t.strip() for t in texts[start:start+max(1,self.max_batch)]];inputs=self._tokenizer(chunk,return_tensors='pt',truncation=True,padding=True,max_length=self.max_input_tokens)
   with torch.inference_mode():output=self._model.generate(**inputs,generation_config=self._generation_config)
   out.extend(text.strip() for text in self._tokenizer.batch_decode(output,skip_special_tokens=True))
  return out
 async def warmup(self):
  if self.enabled:await self.correct_batch(['Проверка запуска корректора.'])
 def metrics(self):return QualityModelStats(self.enabled,self._model is not None,self.model_id,self._calls,self._quantized)
