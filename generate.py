import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = 'runs/p3_official_dev/external_spider/spider_generation_inputs.jsonl'
DEFAULT_OUT = 'runs/p3_official_dev/external_spider'

def fail(message):
    raise SystemExit('STOP: ' + str(message))

def read_jsonl(path):
    if not path.is_file():
        fail(f'文件不存在: {path}')
    with path.open('rb') as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            if not raw.endswith(b'\n'):
                fail(f'JSONL末尾半行，先修复后续跑: {path}:{line_no}')
            try:
                yield json.loads(raw)
            except Exception as exc:
                fail(f'JSONL格式错误: {path}:{line_no}: {exc}')

def schema_for_path(path):
    db = Path(path).resolve()
    if not db.is_file():
        fail(f'找不到SQLite: {db}')
    con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    try:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        ddl = '\n\n'.join((str(r[0]).rstrip(';') + ';' for r in rows if r[0]))
        if not ddl:
            fail(f'空DDL: {db}')
        return ddl
    finally:
        con.close()

def fingerprint(version, model, messages, draw, schema, dbpath):
    obj = {'v': version, 'model': model, 'messages': messages, 'draw': draw, 'db': str(Path(dbpath).resolve()), 'schema': schema, 'max_new': 2048 if model == 'omni' else 1536, 'temperature': 0.7 if model == 'omni' else 0, 'top_p': 0.9 if model == 'omni' else None, 'token_limit': 6144}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def inputs(path):
    rows = list(read_jsonl(path))
    if not rows:
        fail('Spider输入为空')
    ids = [str(x.get('uid', '')) for x in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        fail('Spider UID为空或重复')
    return rows

def worker(args):
    if args.gpu not in (0, 1):
        fail('只能使用GPU 0或1')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from prompts import VERSION, ROOT, official_prompt, make_messages, extract_sql
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    data = inputs(args.inputs)
    tasks = [r for idx, r in enumerate(data) if idx % args.nshards == args.shard]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f'spider_{args.model}_shard{args.shard}.jsonl'
    cache = {}
    for rec in read_jsonl(dest) if dest.exists() else ():
        key = (str(rec['uid']), int(rec['draw']))
        if key in cache:
            fail(f'本分片已有重复候选: {key}')
        cache[key] = rec
    omni_prompt = official_prompt()
    schemas = {}
    todo = []
    for r in tasks:
        dbid = str(r['db_id'])
        dbpath = str(r['db_path'])
        if dbpath not in schemas:
            schemas[dbpath] = schema_for_path(dbpath)
        schema = schemas[dbpath]
        messages = make_messages(args.model, r, schema, omni_prompt)
        for draw in range(5 if args.model == 'omni' else 1):
            fp = fingerprint(VERSION, args.model, messages, draw, schema, dbpath)
            key = (str(r['uid']), draw)
            if key in cache:
                if cache[key]['fingerprint'] != fp:
                    fail(f'源数据/配置改变，旧缓存不可复用: {key}')
            else:
                todo.append((r, draw, messages, fp))
    print(f'[START] model={args.model}, physical GPU={args.gpu}, shard={args.shard}/{args.nshards}, questions={len(tasks)}, completed={len(cache)}, todo={len(todo)}', flush=True)
    if not todo:
        print('[PASS] 无需生成，分片已完整', flush=True)
        return
    model_dir = args.model_path.resolve()
    if not model_dir.is_dir():
        fail(f'本地模型缺失: {model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16), device_map={'': 0}, dtype=torch.float16).eval()
    torch.cuda.reset_peak_memory_stats(0)
    start_all = time.perf_counter()
    with dest.open('a', encoding='utf8') as f:
        for i, (r, draw, msgs, fp) in enumerate(todo, 1):
            tic = time.perf_counter()
            chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            batch = tokenizer(chat, return_tensors='pt')
            input_tokens = batch['input_ids'].shape[1]
            sql, ntokens, status = ('', 0, 'long_context')
            if input_tokens <= 6144:
                batch = {k: v.to('cuda:0') for k, v in batch.items()}
                if args.model == 'omni' and draw > 0:
                    seed = int(hashlib.sha256(f"{VERSION}|{r['uid']}|{draw}".encode()).hexdigest()[:8], 16)
                    torch.manual_seed(seed)
                    opts = {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9}
                else:
                    opts = {'do_sample': False}
                with torch.inference_mode():
                    output = model.generate(**batch, max_new_tokens=2048 if args.model == 'omni' else 1536, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, **opts)
                tokens = output[0][input_tokens:]
                ntokens = len(tokens)
                sql = extract_sql(tokenizer.decode(tokens, skip_special_tokens=True))
                status = 'generated' if sql else 'no_sql'
                del batch, output, tokens
            rec = {'split': 'spider_external', 'uid': r['uid'], 'db_id': r['db_id'], 'model': args.model, 'draw': draw, 'fingerprint': fp, 'sql': sql, 'status': status, 'output_tokens': ntokens, 'generation_seconds': round(time.perf_counter() - tic, 3), 'gpu': args.gpu, 'shard': args.shard}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if i == 1 or i % 25 == 0 or i == len(todo):
                print(f"[{args.model}/gpu{args.gpu} {i}/{len(todo)}] {r['uid']} draw={draw} status={status} elapsed={time.perf_counter() - start_all:.0f}s", flush=True)
    print('[DONE]', dest, 'peak_allocated_GiB=', round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 2), flush=True)

def merge(args):
    data = inputs(args.inputs)
    ids = {r['uid']: r for r in data}
    expect = {(r['uid'], 'omni', j) for r in data for j in range(5)}
    expect |= {(r['uid'], 'qwen', 0) for r in data}
    gathered = {}
    for model, nshards in [('omni', 2), ('qwen', 1)]:
        for shard in range(nshards):
            path = args.out / f'spider_{model}_shard{shard}.jsonl'
            for rec in read_jsonl(path):
                key = (rec['uid'], rec['model'], int(rec['draw']))
                if key in gathered or key not in expect:
                    fail(f'重复或意外候选: {key}')
                gathered[key] = rec
    missing = expect - set(gathered)
    if missing:
        fail(f'候选未生成完整，缺少{len(missing)}条，例如 {sorted(missing)[:4]}')
    problems = [r for r in gathered.values() if not str(r.get('sql') or '').strip()]
    stats = {'questions': len(data), 'candidate_count': len(gathered), 'empty_or_failed': len(problems), 'failed_examples': [[x['uid'], x['model'], x['draw'], x.get('status')] for x in problems[:15]]}
    (args.out / 'spider_generation_summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf8')
    print('[MERGE]', json.dumps(stats, ensure_ascii=False), flush=True)
    if problems:
        print('[WARN] 模型未产出SQL的候选将在评测导出文件中标记为不可执行占位标记，官方EX计0；分片原始日志绝不修改。', flush=True)
    for model in ('omni', 'qwen'):
        target = args.out / f'spider_{model}.jsonl'
        if target.exists():
            fail(f'目标文件已存在，为避免覆盖先人工检查: {target}')
        with target.open('w', encoding='utf8') as f:
            for r in data:
                for draw in range(5 if model == 'omni' else 1):
                    rec = dict(gathered[r['uid'], model, draw])
                    if not str(rec.get('sql') or '').strip():
                        rec['sql'] = '!!MERC_GENERATION_FAILED_NO_SQL!!'
                        rec['evaluation_placeholder'] = True
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print('[MERGED]', target, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['worker', 'merge'])
    p.add_argument('--model', choices=['omni', 'qwen'])
    p.add_argument('--model-path', type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--inputs', type=Path, default=Path(DEFAULT_INPUT))
    p.add_argument('--out', type=Path, default=Path(DEFAULT_OUT))
    args = p.parse_args()
    if args.command == 'worker':
        if args.model_path is None or args.model is None or args.nshards < 1 or (not 0 <= args.shard < args.nshards):
            fail('指定--model并确保--shard在0..nshards-1')
        worker(args)
    else:
        merge(args)
if __name__ == '__main__':
    main()
