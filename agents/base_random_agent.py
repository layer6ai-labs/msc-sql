import os
import random
import constants
import numpy as np
from utils import load_json_file
from schema_utils import generate_schema
from sql_metadata import Parser as SQLParser
from models.embedding_base import DBEmbedder
from sentence_transformers import SentenceTransformer

random.seed(10)
np.random.seed(10)


class BaseRandomAgentWithIndex:
    def __init__(self, index_path):
        if not index_path:
            raise Exception("index_path cannot be empty")
        self.model = None
        self.db_embedder = {}
        self.index_path = index_path
    
    def load_db_index(self, db_id):
        if db_id not in self.db_embedder:
            self.db_embedder[db_id] = DBEmbedder(self.model)
            self.db_embedder[db_id].load_index(self.index_path, db_id)


class BaseBirdAgent(BaseRandomAgentWithIndex):
    def __init__(self, index_path, extra):
        super().__init__(index_path)
        self.model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.db_embedder = {}
        self.index_path = index_path

    def talk(self, message_dict: dict) -> dict:

        db_id = message_dict['db_id']
        self.load_db_index(db_id)

        all_tables = set(message_dict['db_info']['table_info'].keys())

        gt_tables = SQLParser(message_dict['ground_truth']).tables

        remaining_tables = all_tables - set(gt_tables)
        remaining_tables = list(remaining_tables)
        random.shuffle(remaining_tables)

        # extra_n = np.random.choice([0, 1, 2, 3], p=[0.15, 0.3, 0.3, 0.25]) 
        extra_n = 0

        table_subset = gt_tables + remaining_tables[:extra_n]

        generated_schema = generate_schema(
            message_dict['db_id'],
            message_dict['db_info'],
            embedder=self.db_embedder[db_id],
            question=message_dict['query'],
            evidence=message_dict['evidence'],
            include_description=True,
            table_subset=table_subset
        )
       
        # user prompt constructs the prompt that will be used as the user question when fine-tuning the LLM
        # this is different from the variable `prompt` above which is the prompt used for generating the instructions from the LLM.
        user_prompt = constants.bird_generation_user_prompt(
            message_dict['query'],
            message_dict['evidence'],
            generated_schema
        )

        return {
            'user': user_prompt,
            'assistant': message_dict['ground_truth'],
            'extra_tables': remaining_tables[:extra_n]
        }


class BaseBirdFromTablePredAgent(BaseRandomAgentWithIndex):
    def __init__(self, index_path, table_pred_path='output/datasets/instruct_table_selection/report_mistral_table_selection_all_lienar_multipass.json'):
        super().__init__(index_path)
        self.model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.db_embedder = {}
        self.index_path = index_path
        self.table_pred_path = table_pred_path
        self.stage_1_output = load_json_file(table_pred_path)

        self.qid2tables = {}
        for item in self.stage_1_output:
            qid = int(item['idx'].split('_')[1])
            self.qid2tables[qid] = item['tables_pred_union']

    
    def talk(self, message_dict: dict) -> dict:

        db_id = message_dict['db_id']
        self.load_db_index(db_id)

        all_tables = set(message_dict['db_info']['table_info'].keys())

        try:
            table_subset = self.qid2tables[message_dict['idx']]
        except Exception as e:
            print(f"table lookup failed for {message_dict['idx']}, using all tables")
            table_subset = all_tables

        generated_schema = generate_schema(
            message_dict['db_id'],
            message_dict['db_info'],
            embedder=self.db_embedder[db_id],
            question=message_dict['query'],
            evidence=message_dict['evidence'],
            include_description=True,
            table_subset=table_subset
        )
       
        # user prompt constructs the prompt that will be used as the user question when fine-tuning the LLM
        # this is different from the variable `prompt` above which is the prompt used for generating the instructions from the LLM.
        user_prompt = constants.bird_generation_user_prompt(
            message_dict['query'],
            message_dict['evidence'],
            generated_schema
        )

        return {
            'user': user_prompt,
            'assistant': message_dict['ground_truth'],
            'extra_tables': [] 
        }


class BaseSpiderAgent(BaseRandomAgentWithIndex):
    def __init__(self, index_path, extra):
        super().__init__(index_path)
        self.model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.db_embedder = {}
        self.index_path = index_path

    def talk(self, message_dict: dict) -> dict:

        db_id = message_dict['db_id']
        self.load_db_index(db_id)

        all_tables = set(message_dict['db_info']['table_info'].keys())

        gt_tables = SQLParser(message_dict['ground_truth']).tables

        remaining_tables = all_tables - set(gt_tables)
        remaining_tables = list(remaining_tables)
        random.shuffle(remaining_tables)

        # extra_n = np.random.choice([0, 1, 2, 3], p=[0.15, 0.3, 0.3, 0.25]) 
        extra_n = 0

        table_subset = gt_tables + remaining_tables[:extra_n]

        #generated_schema = generate_schema(message_dict['db_id'], message_dict['db_info'], table_subset=table_subset)
        generated_schema = generate_schema(
            message_dict['db_id'],
            message_dict['db_info'],
            embedder=self.db_embedder[db_id],
            question=message_dict['query'],
            evidence=None,
            include_description=False,
            table_subset=table_subset
        )
       
        # user prompt constructs the prompt that will be used as the user question when fine-tuning the LLM
        # this is different from the variable `prompt` above which is the prompt used for generating the instructions from the LLM.
        user_prompt = constants.spider_generation_user_prompt(
            message_dict['query'],
            generated_schema
        )

        return {
            'user': user_prompt,
            'assistant': message_dict['ground_truth'],
            'extra_tables': remaining_tables[:extra_n]
        }


class BaseSpiderFromTablePredAgent(BaseRandomAgentWithIndex):
    def __init__(self, index_path, table_pred_path='output/datasets/instruct_table_selection/report_spider2k_all_tables.json'):
        super().__init__(index_path)
        self.model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.db_embedder = {}
        self.index_path = index_path
        self.table_pred_path = table_pred_path
        self.stage_1_output = load_json_file(table_pred_path)

        self.qid2tables = {}
        for item in self.stage_1_output:
            qid = int(item['idx'].split('_')[1])
            self.qid2tables[qid] = item['tables_pred_union']

    
    def talk(self, message_dict: dict) -> dict:

        db_id = message_dict['db_id']
        self.load_db_index(db_id)

        all_tables = set(message_dict['db_info']['table_info'].keys())

        try:
            table_subset = self.qid2tables[message_dict['idx']]
        except Exception as e:
            print(f"table lookup failed for {message_dict['idx']}, using all tables")
            table_subset = all_tables

        generated_schema = generate_schema(
            message_dict['db_id'],
            message_dict['db_info'],
            embedder=self.db_embedder[db_id],
            question=message_dict['query'],
            evidence=None,
            include_description=False,
            table_subset=table_subset
        )
       
        # user prompt constructs the prompt that will be used as the user question when fine-tuning the LLM
        # this is different from the variable `prompt` above which is the prompt used for generating the instructions from the LLM.
        user_prompt = constants.spider_generation_user_prompt(
            message_dict['query'],
            generated_schema
        )

        return {
            'user': user_prompt,
            'assistant': message_dict['ground_truth'],
            'extra_tables': [] 
        }

