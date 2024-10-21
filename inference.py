from tqdm import tqdm
from utils import load_json_file
from models.embedding_base import DBEmbedder
from sentence_transformers import SentenceTransformer

def index_database(embedder, db_metadata, save_path):
    db_path = db_metadata['db_path']
    db_id = db_metadata['db_id']
    embedder.index_db(db_id, db_path, db_metadata)
    embedder.save_db_index(save_path, db_id)
    embedder.delete_index()


def index(db_metadata, save_path):
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    embedder = DBEmbedder(model)
    index_database(embedder, db_metadata, save_path) 


def get_db_metadata(db_id, db_path):
    ...

class Inference:
    def __init__(self):
        self.stage1_model = ...
        self.stage2_models = [...] 
        self.stage3_model = ...

    # runs stage1 and returns table preds
    def stage1(self, question) -> list[str]:
        ...

    def stage2(self, question):
        ...

    def stage3(self, question):
        ...

    def run(self, db_id, db_path, question, evidence):
        ...

        db_metadata = get_db_metadata(db_id, db_path)
        
