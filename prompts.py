import re,sqlite3,os
from pathlib import Path
ROOT=Path(os.environ.get("MERC_MODEL_ROOT","models")).resolve().parent
VERSION="selector_candidates_v1_20261022"
PROMPT="Task Overview:\nYou are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.\n\nDatabase Engine:\nSQLite\n\nDatabase Schema:\n{db_details}\nThis schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.\n\nQuestion:\n{question}\n\nInstructions:\n- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n- The generated query should return all of the information asked in the question without any missing or extra information.\n- Before generating the final SQL query, please think through the steps of how to write the query.\n\nOutput Format:\nIn your answer, please enclose the generated SQL query in a code block:\n\nTake a deep breath and think step by step to find the correct SQL query."
def official_prompt():
    return PROMPT

def make_messages(model_key, row, schema, omni_prompt):
    question = row['question']
    evidence = str(row.get('evidence') or '')
    if model_key == 'omni':
        if evidence:
            question += '\nExternal Knowledge: ' + evidence
        return [{'role': 'user', 'content': omni_prompt.format(db_details=schema, question=question)}]
    return [{'role': 'system', 'content': 'You are an expert SQLite Text-to-SQL generator. Answer with only an executable SQL query in a SQL code block.'}, {'role': 'user', 'content': f'Database engine: SQLite\nDatabase schema:\n{schema}\n\nQuestion:\n{question}\nExternal knowledge:\n{evidence}\n\nGenerate one correct SQL query. Check joins, predicates, aggregation, sorting, limits, and requested output fields. Use only the real schema. Return only one SQL query in a ```sql fenced code block.'}]

def extract_sql(text):
    text = re.sub('(?is)<think>.*?</think>', '', text or '')
    fences = re.findall('```[ \\t]*(?:sql|sqlite)?[ \\t]*\\r?\\n(.*?)```', text, re.I | re.S)
    for block in fences[::-1] if fences else [text]:
        m = re.search('(?im)^\\s*(SELECT|WITH)\\b', block)
        if not m:
            continue
        sql = block[m.start():].strip()
        acc = ''
        for ch in sql:
            acc += ch
            if ch == ';' and sqlite3.complete_statement(acc):
                return acc.strip()
        if fences:
            return sql
        return re.split('(?im)^\\s*(?:Explanation:|Reasoning:|This query|The query)', sql, maxsplit=1)[0].strip()
    return ''