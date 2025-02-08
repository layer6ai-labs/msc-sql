import os
import torch
import pickle
import sqlite3
import numpy as np
from sentence_transformers.util import cos_sim

class DBEmbedder:
    def __init__(self, model):
        """
        Initializes the Embedder with a SentenceTransformer model.
        
        :param model_name: The name of the SentenceTransformer model to be used.
        """
        # self.model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        self.model = model

        # table -> col -> [values...]  n
        self.indexed_values = {}
        # table -> col -> [[0.2,0.6,-2.6,...]]  nxd
        self.embeddings = {}

    def _index(self, table, col, values: list[str]):
        """
        Indexes the given values by computing their embeddings using the model.
        
        :param values: A list of strings to be indexed.
        """

        if table not in self.indexed_values:
            self.indexed_values[table] = {}
            self.embeddings[table] = {}

        if col in self.indexed_values[table]:
            # already indexed
            return

        self.indexed_values[table][col] = values
        self.embeddings[table][col] = self.model.encode(values, convert_to_tensor=True)

    def index_db(self, db_id, db_path, metadata):
        for table_name, table_data in metadata['table_info'].items():
            for col_name in table_data['col_names']:
                col_info = table_data["col_info"][col_name]
                col_type = col_info["col_type"]

                if col_type.lower() == 'text' or 'varchar' in col_type.lower():
                    if 'varchar' in col_type:
                        print('varchar-------------->', col_type, table_name)
                    print('indexing', db_id, db_path, table_name, col_name)
                    values = self.get_db_values(db_path, table_name, col_name)
                    self._index(table_name, col_name, values)

    def get_db_values(self, db_path, table_name, col_name):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT `{col_name}` FROM \"{table_name}\" LIMIT 2000")
        rows = cursor.fetchall()

        # TODO: Currently limiting the string length of each row val to 200 chars
        values = [row[0] for row in rows]
        values = [val[:200] for val in values if val]
        return values

    def delete_index(self):
        del self.embeddings
        del self.indexed_values
        self.embeddings = {}
        self.indexed_values = {}

    def get_nearest_values(self, query: str, table: str, col: str, k: int) -> list[str]:
        """
        Finds the k nearest values to the query from the indexed values.
        
        :param query: The query string.
        :param table: name of table
        :param col: name of col
        :param k: The number of nearest values to return.
        :return: A list of the k nearest strings to the query.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True).cuda()
        try:
            if self.embeddings[table][col].size()[0] == 0:
                nearest_values = []
            else:
                similarities = cos_sim(query_embedding, self.embeddings[table][col].cuda())[0]
                nearest_indices = torch.argsort(similarities).cpu().numpy()[::-1][:k]
                nearest_values = [self.indexed_values[table][col][i] for i in nearest_indices]
        except Exception as e:
            breakpoint()
            print('nn lookup failed, filling k random values')
            nearest_values = self.indexed_values[table][col][:k]
        return nearest_values

    def save_db_index(self, path, db_id):

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, f'{db_id}_indexed_values.pkl'), 'wb') as f:
            pickle.dump(self.indexed_values, f)

        for table in self.embeddings.keys():
            for col in self.embeddings[table].keys():
                self.embeddings[table][col].cpu()

        with open(os.path.join(path, f'{db_id}_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.embeddings, f)

    def load_index(self, path, db_id):
        with open(os.path.join(path, f'{db_id}_indexed_values.pkl'), 'rb') as f:
            self.indexed_values = pickle.load(f)

        with open(os.path.join(path, f'{db_id}_embeddings.pkl'), 'rb') as f:
            self.embeddings = pickle.load(f)

        for table in self.embeddings.keys():
            for col in self.embeddings[table].keys():
                self.embeddings[table][col].to(device='cuda:0')
