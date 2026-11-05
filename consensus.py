import hashlib,json,sqlite3,time
from pathlib import Path
from collections import Counter
SELECT_TIMEOUT=12
ROW_LIMIT=50000

def norm_value(x):
    if x is None:
        return ['null']
    if isinstance(x, bytes):
        return ['bytes', x.hex()]
    if isinstance(x, (int, float)):
        if isinstance(x, float):
            x = round(x, 8)
        return ['num', str(x)]
    return ['text', str(x)]

def selection_execute(dbpath, sql):
    start = time.monotonic()
    conn = None
    try:
        if not sql.strip():
            return {'ok': False, 'error': 'empty_sql'}
        conn = sqlite3.connect(Path(dbpath).as_uri() + '?mode=ro', uri=True, timeout=1)
        conn.execute('PRAGMA query_only=ON')
        conn.set_progress_handler(lambda: int(time.monotonic() - start > SELECT_TIMEOUT), 1000)
        cursor = conn.execute(sql)
        rows = []
        while True:
            batch = cursor.fetchmany(1024)
            if not batch:
                break
            for row in batch:
                rows.append(json.dumps([norm_value(x) for x in row], ensure_ascii=False, separators=(',', ':')))
            if len(rows) > ROW_LIMIT:
                raise RuntimeError('ROW_LIMIT')
            if time.monotonic() - start > SELECT_TIMEOUT:
                raise TimeoutError('SQL_TIMEOUT')
        signature = hashlib.sha256(json.dumps(sorted(Counter(rows).items())).encode()).hexdigest()
        return {'ok': True, 'sig': signature}
    except Exception as e:
        return {'ok': False, 'error': str(e)[:160]}
    finally:
        if conn is not None:
            conn.close()

def majority(executions, indices):
    pairs = [(executions[i]['sig'], i) for i in indices if executions[i]['ok']]
    if not pairs:
        return indices[0]
    votes = Counter((s for s, _ in pairs))
    maximum = max(votes.values())
    return next((i for s, i in pairs if votes[s] == maximum))

def select(db_path, candidates, composition='4O1Q'):
    if len(candidates) != 6:
        raise ValueError('Expected [o0,o1,o2,o3,o4,a]')
    indices = [0,1,2,3,5] if composition == '4O1Q' else [0,1,2,3,4]
    executions = [selection_execute(str(Path(db_path).resolve()), sql) for sql in candidates]
    index = majority(executions, indices)
    return {'selected_index':index,'sql':candidates[index],'executions':executions}

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',required=True)
    p.add_argument('--db-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--composition',choices=['4O1Q','5O'],default='4O1Q')
    a=p.parse_args()
    root=Path(a.db_root).resolve()
    dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    with dest.open('w',encoding='utf-8') as f:
        for line in Path(a.inputs).read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line);db=(root/r['database_file']).resolve()
            if not db.is_relative_to(root):raise ValueError('database_file must stay within db-root')
            result=select(db,r['sql_candidates'],a.composition)
            f.write(json.dumps({'uid':r['uid'],'db_id':r['db_id'],**result},ensure_ascii=False)+'\n')
if __name__=='__main__':main()
