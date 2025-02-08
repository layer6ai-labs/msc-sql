from enum import Enum

class DATASETS(Enum):
    BIRD_DATASET = "bird"
    SPIDER_DATASET = "spider"


bird_generation_user_prompt = lambda query, evidence, schema: \
f"""You are given an 'SQL Question', 'Evidence' which is information that you need to use to solve the question, 'DB schema' containing the database schema.
Think step by step and solve the question by coming up with the correct SQL statement that solves the question.

Important things to keep in mind:
1. Only use the tables that are required to solve the task.
2. Use the "evidence" and sample values and column descriptions given to you for reasoning.
3. Don't forget keywords such as DISTINCT, WHERE, GROUP BY, ORDER BY, LIMIT, etc. if needed.

SQL Question: {query}

Evidence: {evidence}

DB schema: {schema}

Reminder of the SQL Question: {query}




"""


bird_table_selection_generation_user_prompt = lambda query, evidence, schema: \
f"""As an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. You are given an 'SQL Question', 'Evidence' which is information that you need to use to solve the question. 'DB schema' containing the database schema.
Think step by step. Identify and list all the relevant tables names from the DB schema based on the user question, database schema and evidence provided. Make sure you include all of them.

SQL Question: {query}

Evidence: {evidence}

DB schema: {schema}

"""

bird_generation_agent_prompt = lambda query, evidence, schema, sql: \
f"""You are given an SQL question, evidence which contains more information needed to solve the question, the database schema and the actual SQL statement. 
Think step by step how to EXACLTY arrive at the given SQLITE statement. Make the final answer exactly the same as the given SQL statement. Use the "evidence" given to you for reasoning.
Don't explain the result of the sql statement, but rather provide step by step reasoning to arrive at the solution from the question.
Format the final SQL statement in markdown format.

SQL Question: {query}

Evidence: {evidence}

DB schema: {schema}

SQL statement: {sql}
"""


spider_generation_user_prompt = lambda query, schema: \
f"""You are given an 'SQL Question', 'Evidence' which is information that you need to use to solve the question, 'DB schema' containing the database schema.
Think step by step and solve the question by coming up with the correct SQL statement that solves the question.

Important things to keep in mind:
1. Only use the tables that are required to solve the task.
2. Use the "evidence" and sample values and column descriptions given to you for reasoning.
3. Don't forget keywords such as DISTINCT, WHERE, GROUP BY, ORDER BY, LIMIT, etc. if needed.

SQL Question: {query}

Evidence: 

DB schema: {schema}

Reminder of the SQL Question: {query}

"""


spider_table_selection_generation_user_prompt = lambda query, schema: \
f"""As an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. You are given an 'SQL Question', 'DB schema' containing the database schema.
Think step by step. Identify and list all the relevant tables names from the DB schema based on the user question and database schema provided. Make sure you include all of them.

SQL Question: {query}

DB schema: {schema}

"""


spider_generation_agent_prompt = lambda query, schema, sql: \
f"""You are given an SQL question, the database schema and the actual SQL statement. 
Think step by step how to EXACLTY arrive at the given SQLITE statement. Make the final answer exactly the same as the given SQL statement.
Don't explain the result of the sql statement, but rather provide step by step reasoning to arrive at the solution from the question. 
Format the final SQL statement in markdown format.

SQL Question: {query}

DB schema: {schema}

SQL statement: {sql}
"""

stage3_postamble = lambda q1, q2, r1, r2: \
        f"\n\n The following are the two SQL queries written by the user along with the sample results they generated. One is correct and one is wrong. You need to decide which one is correct.\n\n1: {q1}\nResults of 1st SQL:\n{r1}\n\n2: {q2}\nResults of 2nd SQL:\n{r2}\n\nProvide the number of the right SQL:"

stage3_preamble = """
You are an SQL, database expert. A previous user was given a task of writing a SQL query given a question prompt. 
The user wrote possible SQL queries. One is correct and one is wrong given the question. 
You task is to use the question, results and your expertise to decide which one is correct. 

Here is the question prompt: 

"""

stage3_post_2_sql =lambda q1, q2, result1, result2: \
    f"""
        The following are the SQL queries written by the user.
        Only one is correct. You need to decide which one is correct.
        
        1){q1}
        
        With an output of:
        
        {result1}
        
        2){q2}

        With an output of: 
        
        {result2}

        Reminder:

        SQL 1) {q1}
        SQL 2) {q2}

        Based on the SQL Query and resulting output, analyze the which is correct. Pay attention to the question and evidence, think step-by-step and reason through the SQL query and results.
        Provide the number of the of the correct SQL:
        """
