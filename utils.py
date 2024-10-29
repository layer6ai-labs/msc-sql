import os
import re
import glob
import json
import sqlite3
from typing import Dict


def is_valid_date(date_str):
    if (not isinstance(date_str, str)):
        return False
    date_str = date_str.split()[0]
    if len(date_str) != 10:
        return False
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date_str):
        year, month, day = map(int, date_str.split('-'))
        if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
            return False
        else:
            return True
    else:
        return False


def is_valid_date_column(col_value_lst):
    for col_value in col_value_lst:
        if not is_valid_date(col_value):
            return False
    return True


def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


def extract_world_info(message_dict: dict):
    info_dict = {}
    info_dict['idx'] = message_dict['idx']
    info_dict['db_id'] = message_dict['db_id']
    info_dict['query'] = message_dict['query']
    info_dict['evidence'] = message_dict.get('evidence', '')
    info_dict['difficulty'] = message_dict.get('difficulty', '')
    info_dict['ground_truth'] = message_dict.get('ground_truth', '')
    info_dict['send_to'] = message_dict.get('send_to', '')
    return info_dict


def replace_multiple_spaces(text):
    pattern = r'\s+'
    new_text = re.sub(pattern, ' ', text)
    return new_text


# SQL parsing
def extract_table_names(sql_query: str):
    sql_query = sql_query.replace('`', '')
    table_names = re.findall(r'FROM\s+([\w]+)', sql_query, re.IGNORECASE) + \
                  re.findall(r'JOIN\s+([\w]+)', sql_query, re.IGNORECASE)
    return set(table_names)


def get_used_tables(sql, db_path) -> dict:  # table_name -> chosen columns & discarded columns
    table_names = extract_table_names(sql)
    sch = {}
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = cursor.fetchall()
        column_names = [cinfo[1] for cinfo in columns]
        sch[table_name] = {
            "chosen columns": column_names,
            "discarded columns": []
        }
    return sch


def get_all_tables(db_path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type=\'table\'")
    tables = cursor.fetchall()
    table_names = [a[0] for a in tables if a[0] != 'sqlite_sequence']
    sch = {}
    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = cursor.fetchall()
        column_names = [cinfo[1] for cinfo in columns]
        sch[table_name] = {
            "chosen columns": column_names,
            "discarded columns": []
        }
    return sch


def get_files(root, suffix):
    """
        returns absolute path of all files in a director 'root' with a suffix recursively
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f'path {root} not found.')
    res = glob.glob(f'{root}/**/*{suffix}', recursive=True)
    res = [os.path.abspath(p) for p in res]
    return res


# read txt file to string list and strip empty lines
def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"load txt file from {path}")
        return [line.strip() for line in f if line.strip()!= '']


def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"load json file from {path}")
        return json.load(f)


def load_jsonl_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            js_str = line.strip()
            if js_str == '':
                continue
            js = json.loads(js_str)
            data.append(js)
        print(f"load jsonl file from {path}")
        return data


def save_file(path, string_lst):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(string_lst)
        print(f"save file to {path}")


def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"save json file to {path}")


def save_jsonl_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for js in data:
            f.write(json.dumps(js, ensure_ascii=False) + '\n')
        print(f"save jsonl file to {path}")


def convert_json_to_jsonl(input_path, output_path):
    data = load_json_file(input_path)
    save_jsonl_file(output_path, data)


# check if valid format
def check_selector_response(json_data: Dict) -> bool:
    FLAGS = ['keep_all', 'drop_all']
    for _, v in json_data.items():
        if isinstance(v, str):
            if v not in FLAGS:
                print(f"error: invalid table flag: {v}\n")
                print(f"json_data: {json_data}\n\n")
                return False
        elif isinstance(v, list):
            pass
        else:
            print(f"error: invalid flag type: {v}\n")
            print(f"json_data: {json_data}\n\n")
            return False
    return True


def parse_json(text: str) -> dict:
    start = text.find("```json")
    end = text.find("```", start + 7)
    
    if start != -1 and end != -1:
        json_string = text[start + 7: end]
        
        try:
            json_data = json.loads(json_string)
            valid = check_selector_response(json_data)
            if valid:
                return json_data
            else:
                return {}
        except:
            print(f"error: parse json error!\n")
            print(f"json_string: {json_string}\n\n")
            pass
    
    return {}


def parse_sql(res: str) -> str:
    """Only need SQL(startswith `SELECT`) of LLM result"""
    if 'SELECT' not in res and 'select' not in res:
        res = 'SELECT ' + res
    # match = re.search(parse_pattern, res, re.IGNORECASE | re.DOTALL)
    # if match:
    #     sql = match.group().strip()
    #     sql = sql.replace('```', '') # TODO
    #     sql = sql.replace('\n', ' ') # TODO
    #     return True, sql
    # else:
    #     return False, ""
    res = res.replace('\n', ' ')
    return res.strip()


def parse_single_sql(res: str) -> str:  # if do not need decompose, just one code block is OK!
    """Return SQL in markdown block"""
    lines = res.split('\n')
    iter, start_idx, end_idx = -1, -1, -1
    for idx in range(iter + 1, len(lines)):
        if '```' in lines[idx]:
            start_idx = idx
            break
    if start_idx == -1: return ""
    for idx in range(start_idx + 1, len(lines)):
        if '```' in lines[idx]:
            end_idx = idx
            break
    if end_idx == -1: return ""

    return " ".join(lines[start_idx + 1: end_idx])


def extract_sql_from_markdown(markdown):
    # Define the regular expression pattern to match SQL strings
    sql_pattern = r'```sql\s*([\s\S]+?)```'
    sql_matches = re.findall(sql_pattern, markdown, re.IGNORECASE)

    if not sql_matches:
        sql = parse_sql(markdown)
    else:
        sql = parse_sql(sql_matches[0])
    return sql


def extract_last_sql_block(text):
    # Use regular expression to find all sql blocks
    sql_blocks = re.findall(r'```sql[ ]*\n(.*?)```', text, re.DOTALL)

    # If there are no sql blocks, return None
    if not sql_blocks:
        return None

    # Return the contents of the last sql block
    return sql_blocks[-1]


def extract_table_name_list(text):
    try:
        table_names = text.split(', ')
    except:
        # Wrong format
        return None
    # print(table_names, text)
    return table_names

def add_prefix(sql):
    if not sql.startswith('SELECT') and not sql.startswith('select'):
        sql = 'SELECT' + sql
    return sql
