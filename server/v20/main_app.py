"""Recommended profile: proven local hybrid + syntax/RAG guards, bounded reasoning."""
import os
os.environ.setdefault('LLM_PRESET','Z')
os.environ.setdefault('REASONING_MODE','coverage')
os.environ.setdefault('REASONING_TOTAL_TIMEOUT','20')
os.environ.setdefault('RAG_ENABLED','true')
from decision_app_v12 import app
app.title='Maincheck v20 main: hybrid-syntax-rag'
app.version='20-main'
