from tqdm import tqdm
from utils import load_json_file
from models.embedding_base import DBEmbedder
from sentence_transformers import SentenceTransformer

import stage2_generate
import yaml


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
    pass

class Inference:
    def __init__(self, config_file):
        # Load the YAML configuration
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.stage2_models = config['stage2_models']

    def stage1(self, input_file) -> list[str]:
        pass

    def stage2(self, stage1_output_file):
        print("Running Stage 2 Inference")
        for model in tqdm(self.stage2_models):
            stage2_generate.stage_2_pipeline(
                model_name=model['model_name'],
                peft_model=model['peft_model'],
                eval_ds_path=stage1_output_file,
                eval_batch_size=1,
                dataset_name='bird',
                eval_percent=0.005,
                intermediate_jsonl_results_file=model['intermediate_jsonl_results_file'],
                final_json_results_file=model['final_json_results_file']
            )


    def stage3(self, stage2_output_files):
        pass

    def run(self, db_id, db_path, question, evidence):
        

        db_metadata = get_db_metadata(db_id, db_path)
        

if __name__ == "__main__":
    inference = Inference('inference_config.yaml')
    # inference.stage1(...)
    inference.stage2('/home/ilan/sqllm/dev_sept_13_rag_new_mistral1_with_comment.json')
    # inference.stage3(...)