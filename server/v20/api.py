from __future__ import annotations
import time
from typing import Protocol
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import PlainTextResponse,JSONResponse
from .protocol import render

class Engine(Protocol):
    name:str
    async def correct(self,text:str,context:str): ...
    async def health(self)->dict: ...

def make_app(engine:Engine)->FastAPI:
    app=FastAPI(title=f"Maincheck v20 {engine.name}",version="20.0")
    stats={"calls":0,"failures":0,"latency_ms":[]}
    @app.get('/health')
    async def health():
        state=await engine.health(); code=200 if state.get('ready') else 503
        return JSONResponse(state,status_code=code)
    @app.get('/metrics')
    async def metrics():
        vals=stats['latency_ms']; return {"engine":engine.name,"calls":stats['calls'],"failures":stats['failures'],"last_ms":vals[-1] if vals else 0}
    @app.post('/suggest',response_class=PlainTextResponse)
    async def suggest(text:UploadFile=File(...),context:UploadFile=File(...)):
        raw=(await text.read()).decode('utf-8','replace').replace('\r\n','\n').replace('\r','\n').strip(); ctx=(await context.read()).decode('utf-8','replace').strip()
        started=time.perf_counter(); stats['calls']+=1
        try:
            corrected,edits=await engine.correct(raw,ctx); return render(corrected,edits)
        except Exception as exc:
            stats['failures']+=1; return PlainTextResponse(f"ОШИБКА_СЕРВЕРА: {type(exc).__name__}: {exc}",status_code=500)
        finally:
            stats['latency_ms'].append(int((time.perf_counter()-started)*1000)); stats['latency_ms'][:]=stats['latency_ms'][-200:]
    return app
