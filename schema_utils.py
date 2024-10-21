import sqlite3
import random
random.seed(42)


def generate_schema(db_id, metadata, embedder=None, question=None, evidence=None, include_description=False, table_subset=None, num_example_rows=0, num_simple_example_values=3):
    """
    Generate the SQL schema based on the provided metadata.
    Args:
        metadata (dict): The metadata containing information about the tables and columns.
        include_description (bool, optional): Whether to include column descriptions in the schema. Defaults to False.
        table_subset (list, optional): A list of table names to include in the schema. If None, all tables will be included.
    Returns:
        str: The generated SQL schema.
    """


    if table_subset:
        table_subset = [table.lower().strip() for table in table_subset]


    # schema = f"Database: {db_id}\n"
    schema = f"{db_id}\n"

    for table_name, table_data in metadata["table_info"].items():

        if table_subset and table_name.lower() not in table_subset:
            continue

        schema += f"TABLE \"{table_name}\"\n(\n"
        for col_name in table_data["col_names"]:
            col_info = table_data["col_info"][col_name]
            col_type = col_info["col_type"]
            col_desc = col_info["description"]


            schema += f" {format_col(col_name)} {col_type}"
            if col_info["is_pkey"]:
                schema += " primary key"
            schema += ","

            if include_description or num_simple_example_values:
                # adding comment
                schema += " -- "

            if include_description and col_desc and type(col_desc) == str:
                if len(col_desc) > 0 and col_desc != col_name:
                    schema += f"{col_desc}"


            if num_simple_example_values > 0:
                # schema += get_sample_col_values(metadata["db_path"], table_name, col_name, num_simple_example_values)
                assert question is not None, "provide question"
                schema += get_similar_col_values(question, evidence, db_id, metadata['db_path'], table_name, col_name, col_type, embedder, num_simple_example_values)

            schema += "\n"
        pk_columns = table_data["primary_keys"]
        if len(pk_columns) > 1:
            schema += " primary key ("
            schema += ", ".join(pk_columns)
            schema += ")\n"
        for foreign_key_relationship in metadata["foreign_key_relationships"]:
            if table_name == foreign_key_relationship[0][0]:
                column = format_col(foreign_key_relationship[0][1])
                referenced_table = foreign_key_relationship[1][0]
                referenced_column = format_col(foreign_key_relationship[1][1])
                if len(referenced_column) > 0:
                    schema += f" foreign key ({column}) references {referenced_table}({referenced_column})\n"
                else:
                    schema += f" foreign key ({column}) references {referenced_table}\n"
        schema += ")\n\n"


        if num_example_rows > 0:
            db_path = metadata["db_path"]
            # Load the sqlite file and extract the first 3 rows as a string to attach it to the schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT {num_example_rows}")
            rows = cursor.fetchall()
            schema += f"Sample rows from {table_name}:\n"
            for row in rows:
                for col in row:
                    if type(col) == str and len(col) > 15:
                        schema += f"\"{col[:30]}...\", "
                    elif type(col) == str:
                        schema += f"\"{col}\", "
                    else:
                        schema += f"{col}, "
                schema += "\n"
            schema += "\n"

            conn.close()

    return schema


def format_col(col):
    if " " in col or "-" in col:
        return f'"{col}"'
    return col


def get_sample_col_values(db_path, table_name, col_name, num_simple_example_values):
    # Load the sqlite file and extract the first 3 rows as a string to attach it to the schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT DISTINCT `{col_name}` FROM \"{table_name}\" LIMIT {num_simple_example_values}")
    rows = cursor.fetchall()
    sample_string = f" Example: "
    for row in rows:
        col = row[0]
        if type(col) == str and len(col) > 15:
            sample_string += f"\"{col[:15]}...\", "
        elif type(col) == str:
            sample_string += f"\"{col}\", "
        elif type(col) == float:
            # if float, keep up to 3 decimal places
            sample_string += f"{col:.3f}, "
        else:
            sample_string += f"{col}, "
    conn.close()
    return sample_string


def get_similar_col_values(question, evidence, db_id, db_path, table_name, col_name, col_type, embedder, topk):
    """
        load values from the table, if the table col type is text, get semantically similar "topk" values.
        Else for date, real and integer types, get first 3 values
    """

    if not embedder:
        raise Exception("Embedder cannot be None")

    sample_string = f" Example: "

    # get nearest k with nn search based only on 
    if col_type.lower() == 'text' or 'varchar' in col_type.lower():

        if not evidence:
            query = question
        else:
            query = f"{question}\n{evidence}"

        closest_values = embedder.get_nearest_values(query, table_name, col_name, topk)

        for col in closest_values:
            if type(col) == str and len(col) > 15:
                if "http" in col[:5]:
                    sample_string += f'"{col[:7]}...", '
                else: 
                    sample_string += f'"{col[:20]}...", '
            elif type(col) == str:
                sample_string += f'"{col}", '

    else:

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT DISTINCT `{col_name}` FROM \"{table_name}\" LIMIT {topk}")
        rows = cursor.fetchall()
        values = [row[0] for row in rows]
        for col in values:
            if type(col) == float:
                # if float, keep up to 3 decimal places
                sample_string += f"{col:.3f}, "
            else:
                sample_string += f"{col}, "
    
        conn.close()

    return sample_string
