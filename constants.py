from enum import Enum

class DATASETS(Enum):
    BIRD_DATASET = "bird"
    SPIDER_DATASET = "spider"


bird_generation_user_prompt_ilan = lambda query, evidence, schema: \
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


spider_generation_user_prompt_ilan = lambda query, schema: \
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


spider_generation_agent_prompt = lambda query, schema, sql: \
f"""You are given an SQL question, the database schema and the actual SQL statement. 
Think step by step how to EXACLTY arrive at the given SQLITE statement. Make the final answer exactly the same as the given SQL statement.
Don't explain the result of the sql statement, but rather provide step by step reasoning to arrive at the solution from the question. 
Format the final SQL statement in markdown format.

SQL Question: {query}

DB schema: {schema}

SQL statement: {sql}
"""

stage3_preamble = "You are an SQL, database expert. A previous user was given a task of writing a SQL query given a question prompt. The user wrote 2 possible SQL queries. One is correct and one is wrong given the question. You task is to use the question and your expertise to decide which one is correct. Here is the question prompt: \n\n"

stage3_postamble = lambda q1, q2, r1, r2: \
        f"\n\n The following are the two SQL queries written by the user along with the sample results they generated. One is correct and one is wrong. You need to decide which one is correct.\n\n1: {q1}\nResults of 1st SQL:\n{r1}\n\n2: {q2}\nResults of 2nd SQL:\n{r2}\n\nProvide the number of the right SQL:"
