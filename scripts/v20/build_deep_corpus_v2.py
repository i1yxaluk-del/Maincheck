#!/usr/bin/env python3
"""Build deep corpus v2 with an unambiguous participial-clause case."""
import argparse,json
from pathlib import Path
from build_deep_corpus import P

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
 rows=[dict(row) for row in P]
 case=next(row for row in rows if row['id']=='deep-participle')
 case['text']='Документы поступившие вчера из отдела зарегистрированы в журнале входящей корреспонденции.'
 case['expected_text']='Документы, поступившие вчера из отдела, зарегистрированы в журнале входящей корреспонденции.'
 target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
 print(f'DEEP CORPUS V2 BUILT cases={len(rows)} positives={sum(not x["clean"] for x in rows)} clean={sum(x["clean"] for x in rows)} out={target}')
if __name__=='__main__':main()
