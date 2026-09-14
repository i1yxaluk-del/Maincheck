"""Устойчивый клиент эмбеддингов Ollama: пакетный /api/embed и legacy fallback."""
from __future__ import annotations
import os,time
from typing import List,Optional

class OllamaEmbedder:
    def __init__(self,model="nomic-embed-text",base_url="http://localhost:11434",timeout=120.0):
        self.model=model;self.base_url=base_url.rstrip("/");self.timeout=timeout
        self.name=f"ollama:{model}";self._dim:Optional[int]=None
        self.batch_size=max(1,int(os.getenv("RAG_EMBED_BATCH","8")))
        self.retries=max(1,int(os.getenv("RAG_EMBED_RETRIES","3")))
    @property
    def dim(self):
        if self._dim is None:self._dim=len(self.embed(["probe"])[0])
        return self._dim
    def _batch(self,client,texts):
        response=client.post(self.base_url+"/api/embed",json={"model":self.model,"input":texts,"truncate":True,"keep_alive":"15m"})
        if response.status_code==404:return None
        if response.is_error:raise RuntimeError(f"Ollama /api/embed: HTTP {response.status_code}: {response.text[:1000]}")
        vectors=response.json().get("embeddings") or []
        if len(vectors)!=len(texts):raise RuntimeError(f"Ollama вернул {len(vectors)} векторов вместо {len(texts)}")
        return vectors
    def _legacy(self,client,texts):
        vectors=[]
        for text in texts:
            response=client.post(self.base_url+"/api/embeddings",json={"model":self.model,"prompt":text,"keep_alive":"15m"})
            if response.is_error:raise RuntimeError(f"Ollama /api/embeddings: HTTP {response.status_code}: {response.text[:1000]}")
            vector=response.json().get("embedding") or []
            if not vector:raise RuntimeError("Ollama вернул пустой эмбеддинг")
            vectors.append(vector)
        return vectors
    def embed(self,texts:List[str])->List[List[float]]:
        import httpx
        result=[]
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0,len(texts),self.batch_size):
                batch=texts[start:start+self.batch_size];last=None
                for attempt in range(self.retries):
                    try:
                        vectors=self._batch(client,batch)
                        if vectors is None:vectors=self._legacy(client,batch)
                        result.extend(vectors);last=None;break
                    except Exception as error:
                        last=error
                        if attempt+1<self.retries:time.sleep(2**attempt)
                if last is not None:raise RuntimeError(f"Не удалось получить эмбеддинги после {self.retries} попыток: {last}")
        if result:self._dim=len(result[0])
        return result
