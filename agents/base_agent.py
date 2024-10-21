from models.embedding_base import DBEmbedder

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


