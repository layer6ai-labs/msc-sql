import os
import sqlite3
import argparse
import pandas as pd
from tqdm import tqdm
from utils import load_json_file, save_json_file

BIRD_DATASET = "bird"
SPIDER_DATASET = "spider"
ARCHER_DATASET = "archer"


def get_col_description(db_id, db_prefix_path, table_name):
    """
        reads column description for a table in a given db and returns them as a list in the order 
        specified in the tables json file
    """
    descr_file = os.path.join(db_prefix_path, db_id, 'database_description', f'{table_name}.csv')
    ret_val = []
    try:
        df = pd.read_csv(descr_file)
        ret_val = list(df["column_description"])
    except UnicodeDecodeError as uni_err:
        df = pd.read_csv(descr_file, encoding="unicode_escape")
        ret_val = list(df["column_description"])
    except Exception as e:
        found = False
        if found:
            descr_file = os.path.join(db_prefix_path, db_id, 'database_description', f'{table_name}.csv')
            try:
                df = pd.read_csv(descr_file)
            except Exception as inner_e:
                print("exception:", inner_e)
                df = pd.read_csv(descr_file, encoding="unicode_escape")
            ret_val = list(df["column_description"])

        else:
            print(f"WARNING: no such table or column: {table_name}, in db: {db_id} located at: {descr_file}", e)

    return [item if item else '' for item in ret_val]


def construct_db_table_names_dict(table_data):
    """
        Returns a dictionary with a list of table names for each db_id
        {
            <db_id>: {
                'table_names': [...],
            },
            <db_id>: ...
        }
    """
    table_dict = {}
    for item in table_data:
        db_id = item["db_id"]
        table_names = item["table_names"]
        table_dict[db_id] = {'table_names': table_names}  

    return table_dict

def construct_foreign_keys_from_metadata(table_data):
    """
        From the table metadata file, extract the foreign key relationships
        This is not currently being used as the column names are often misspelled. 
        Returns foreign key relationships in the following format:
        {
            <db_id>: {
                'table_names': [...],
                'foreign_key_relationships': [
                        (<from_table>, <from_col>), (<to_table>, <to_col>),
                        ...
                ]
            },
            ...
        }
    """
    fkey_dict = {}
    for item in table_data:
        db_id = item["db_id"]
        table_names = item["table_names"]
        fkey_dict[db_id] = {'table_names': table_names, 'foreign_key_relationships': []}
        
        # for each col_idx (in item['column_names']) we want to construct a map col_idx -> (name of the col, table it came from)
        col_names_and_table_dict = {} 
        for col_idx, col_name in enumerate(item['column_names']):
            # col_name[0] gives the table idx
            if col_name[0] == -1:
                continue              
            col_names_and_table_dict[col_idx] = (col_name[1], table_names[col_name[0]])
    
        for a, b in item['foreign_keys']:
            a_col, a_tab = col_names_and_table_dict[a]
            b_col, b_tab = col_names_and_table_dict[b]
            fkey_dict[db_id]['foreign_key_relationships'].append([(a_tab, a_col),(b_tab, b_col)])

    return fkey_dict


def extract_tables_columns_and_pks(dataset, db_id, db_prefix_path, table_names_from_json_file):
    """
        known issues: 
        a few table primary keys were missing in the dev set when using PRAGMA table_info
        table names in database don't exactly match table names in the tables.json,
            we have to use table_name in tables.json to read from dev_databases/<db>/database_descriptions/<table_name>.csv/ and not from the db.
        names of columns in db isn't the same as the ones specified in tables.json (both original_column_names and column_names fields).
        
        The tables.json file specifies primary keys as index values of the list specified in column_names.
        Similary the tables.json file specifies foreign key relationships with the index values of the list specified in 'column_names'.

        I am unable to extract foreign key relationships of all databases using PRAGMA foreign_key_list;
        primary keys can be extracted using PRAGMA table_info(`table_name`);

        Retaining exact table, column names and primary key cols from db file instead of using table.json.
        The foreign keys are also extracted from the db file. Important to note that if the column name
        is not defined for the foreign key reference column, then it corresponds to (one of) the primary key column.

        returns info on all tables in the db in the below format:
        {
            <table_name>: {
                'col_info': {
                    <col_name>: {
                        'col_type': TEXT,
                        'description': ...,
                        'is_pkey': True/False
                    },
                    ...
                },
                'primary_keys': [...],
            },
            ...
        }
        and the foreign key relationships in the below format:
        {
            [
                [(from_table, from_col), (to_table, to_col)],
                ...
            ]
        }
            
    """

    data = {}
    db_path = os.path.join(db_prefix_path, db_id, f'{db_id}.sqlite')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get information about all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [tab[0] for tab in tables if tab[0] != 'sqlite_sequence']

    fkey_data = []

    for tab_idx, table_name in enumerate(tables):
        primary_keys = []
        all_columns = []
        data[table_name] = {'col_info': {}}
        cursor.execute("PRAGMA table_info('{}');".format(table_name))
        table_info = cursor.fetchall()

       
        if dataset == BIRD_DATASET:
            # getting column descriptions for the BIRD dataset
            # firstly: using table names from json file
            # secondly the file: 'data/bird/dev/dev_databases/{db_id}/database_description/{tab_name}.csv' seems to have badly written names that don't match the names in the db (with extra spaces, diff casing etc)
            # but the order seems to be the same as it appears in table columns, hence getting column descriptions and assigning the description to the column in that position. Hoping for the best!
            col_descriptions = get_col_description(db_id, db_prefix_path, table_names_from_json_file[tab_idx])
        else:
            col_descriptions = []

        for col_idx, column_info in enumerate(table_info):
            # example col_info: (0, 'CustomerID', 'INTEGER', 1, None, 1)
            col_name = column_info[1] # 1st column holds the column name
            col_type = column_info[2] # 2st column holds the column type
            all_columns.append(col_name)
            
            if (column_info[-1] != 0):  # 5th column non-zero value represents whether a column is part of the primary key
                primary_keys.append(column_info[1])
            data[table_name]['col_info'][col_name] = {'col_type': col_type, 'description': col_descriptions[col_idx] if col_idx < len(col_descriptions) else '', 'is_pkey': column_info[-1] != 0}

        data[table_name]['primary_keys'] = primary_keys
        data[table_name]['col_names'] = all_columns

    for tab_idx, table_name in enumerate(tables):

        # get information about foreign key relationships
        foreign_keys_list = []
        cursor.execute("PRAGMA foreign_key_list('{}');".format(table_name))
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            from_table = table_name
            from_col = fk[3]
            to_table = fk[2]
            to_col = fk[4]
            if to_table not in data:
                print(f"WARNING: no such table name: {to_table}")
            else:
                if to_col not in data[to_table]['col_info']:
                    if to_col is not None:
                        print(f"WARNING: no such column: {to_col} in table: {to_table} in db: {db_id}")
                    to_col = ""
                    # Sometimes the to_col is not specified which implies it is the primary key of the to_table
                    # It is possible to fill that in, however, sometimes if the table has a primary key with multiple columns, 
                    # you need to build the foreign relationships using multiple columns, not just 1. 

                    # to_col = data[to_table]['primary_keys'][0]  # Use the first primary key as a fallback
                    # if len(data[to_table]['primary_keys']) > 1:
                    #     print(f"WARNING: Assigning foreign keys:  multiple primary keys in table: {to_table} in db: {db_id}")
                    #     print(" ---- > ", foreign_keys)
                foreign_keys_list.append([(from_table, from_col), (to_table, to_col)])
        
        fkey_data = fkey_data + foreign_keys_list


    conn.close()
    return data, fkey_data

def process_all_dbs(dataset, tables_json_path, db_prefix_path):
    table_data = load_json_file(tables_json_path)
    preprocessed_db = {}
    db_dict = construct_db_table_names_dict(table_data) 
    for db_id in tqdm(db_dict):
        preprocessed_db[db_id] = {}
        db_path = os.path.join(db_prefix_path, db_id, f'{db_id}.sqlite')
        table_info, fk_info = extract_tables_columns_and_pks(dataset, db_id, db_prefix_path, db_dict[db_id]['table_names'])
        preprocessed_db[db_id]['table_info'] = table_info
        preprocessed_db[db_id]['db_path'] = db_path 
        preprocessed_db[db_id]['foreign_key_relationships'] = fk_info 

        
    return preprocessed_db


def process_archer_dataset(db_prefix_path='data/archer/databases'):
    db_names = os.listdir(db_prefix_path)
    #db_paths = [f"{db}.sqlite" for db in db_names]
    train_dbs = ['bike_1', 'customers_and_products_contacts', 'driving_school', 'formula_1', 'hospital_1', 'riding_club', 'soccer_1', 'wine_1']
    dev_dbs = ['concert_singer', 'world_1']

    all_dbs = []
    for db_names in [train_dbs, dev_dbs]:
        preprocessed_db = {}
        for db_id in db_names:
            preprocessed_db[db_id] = {}
            db_path = os.path.join(db_prefix_path, db_id, f'{db_id}.sqlite')
            table_info, fk_info = extract_tables_columns_and_pks(ARCHER_DATASET, db_id, db_prefix_path, None)
            preprocessed_db[db_id]['table_info'] = table_info
            preprocessed_db[db_id]['db_path'] = db_path 
            preprocessed_db[db_id]['foreign_key_relationships'] = fk_info 
        
        all_dbs.append(preprocessed_db)

    return all_dbs[0], all_dbs[1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[BIRD_DATASET, SPIDER_DATASET, ARCHER_DATASET],
        help="The dataset to process(ex: bird, spider)",
    )
    parser.add_argument(
        "--db_prefix_path",
        type=str,
        required=True,
        help="prefix path of db ex: (data/bird/dev/dev_databases/, or data/bird/train/train_databases/)",
    )
    parser.add_argument(
        "--tables_json_path",
        type=str,
        required=True,
        help="path to tables.json (dev_tables.json)",
    )
    parser.add_argument(
        "--out_metadata_path",
        type=str,
        required=True,
        help="path to processed db metadata ex: data/bird/dev/dev_metadata.json",
    )
    args = parser.parse_args()

    if args.dataset == ARCHER_DATASET:
        train_processed_db_metadata, dev_processed_db_metadata = process_archer_dataset(args.db_prefix_path)
        save_json_file('data/archer/train_metadata.json', train_processed_db_metadata)
        save_json_file('data/archer/dev_metadata.json', dev_processed_db_metadata)
    else:
        processed_db_metadata = process_all_dbs(args.dataset, args.tables_json_path, args.db_prefix_path)
        save_json_file(args.out_metadata_path, processed_db_metadata)
