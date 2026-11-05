import re,sys,sqlite3
from pathlib import Path
def official_judge(job):
    key, index, sql, gold, database, timeout = job
    if not re.match('^\\s*(SELECT|WITH)\\b', sql, re.I):
        return {'key': key, 'correct': 0, 'status': 'not_readonly_query'}
    sys.path.insert(0, str(OFFICIAL.resolve()))
    import evaluation_utils
    import evaluation_ex

    def readonly_connect(dialect, path):
        if dialect != 'SQLite':
            raise RuntimeError('Only SQLite is supported')
        return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    evaluation_utils.connect_db = readonly_connect
    result = evaluation_ex.execute_model(sql, gold, database, index, timeout, 'SQLite')
    return {'key': key, 'correct': int(result['res']), 'status': 'official_execute_model'}

def main():
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--evaluator-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    global OFFICIAL
    OFFICIAL=Path(a.evaluator_dir).resolve()
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    with dest.open('w',encoding='utf-8') as f:
        for line in Path(a.inputs).read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line);res=official_judge((r['uid'],0,r['sql'],r['gold_sql'],str(Path(r['db_path']).resolve()),30));f.write(json.dumps({'uid':r['uid'],**res})+'\n')
if __name__=='__main__':main()
