from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[2];path=ROOT/'server/cloud/main.py';source=path.read_text(encoding='utf-8');tree=ast.parse(source)
suggest=None
for node in ast.walk(tree):
 if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='suggest':suggest=node;break
assert suggest is not None
args={a.arg for a in suggest.args.args};assert {'text','context'}<=args
assert '===CORRECTED===' in source and '===CHANGES===' in source and '===END===' in source
imports=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
 elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
assert not any(name.startswith(('decision_engine','legal_style_stage','v20.')) for name in imports)
print('V20 CLOUD CONTRACT OK: multipart text/context and response protocol preserved')
