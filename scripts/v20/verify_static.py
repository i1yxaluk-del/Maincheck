from pathlib import Path
import argparse,ast
ROOT=Path(__file__).resolve().parents[2]
def require(paths):
 for x in paths:
  if not (ROOT/x).exists(): raise SystemExit(f'missing {x}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--services',action='store_true');p.add_argument('--docs',action='store_true');p.add_argument('--ci',action='store_true');a=p.parse_args()
 if a.services:
  files=list((ROOT/'server/local').glob('v20-*.service')); assert len(files)==4
  text='\n'.join(x.read_text() for x in files); assert 'v20.main_app:app' in text and 'v20.openvino_app:app' in text and 'v20.llama_json_app:app' in text; print('V20 SERVICES OK');return
 if a.docs: require(['docs/v20/README.md','docs/v20/ARCHITECTURE.md']); text=(ROOT/'docs/v20/README.md').read_text(); assert all(x in text for x in ('main','openvino','llama-json','rollback')); print('V20 DOCS OK');return
 if a.ci: text=(ROOT/'.github/workflows/v20-engine-lab.yml').read_text(); assert 'verify_static.py' in text and 'benchmark --self-test' in text; print('V20 CI OK');return
 for f in (ROOT/'server/v20').glob('*.py'):ast.parse(f.read_text())
 require(['scripts/v20/install.sh','scripts/v20/install-worker.sh','scripts/v20/benchmark.sh','tests/v20/cases.jsonl'])
 print('V20 STATIC OK')
if __name__=='__main__':main()
