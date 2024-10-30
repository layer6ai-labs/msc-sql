from tqdm import tqdm
from utils import load_json_file
from models.embedding_base import DBEmbedder
from sentence_transformers import SentenceTransformer

import stage1_prediction
import stage2_generate
import stage3_workflow
import yaml
import os
import run_eval

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

        self.config = config

    def stage1(self):
        print('running stage1 pipeline')
        stage1_prediction.stage_1_pipeline(
                model_name="table_selection/mistral_7b_table_selection_0614_all_linear/", # "mistral_7b_schema_linking",
                peft_model=True,
                eval_ds_path=self.config['stage1_input_file'],
                eval_batch_size=2,
                dataset_name='bird',
                eval_percent=0.005,
                intermediate_jsonl_results_file=self.config['stage1_temp_file'],
                final_json_results_file=self.config['stage1_output_file']
            )
        
    def stage2(self):

        # Using the stage1 output file, run stage2 inference
        stage1_output_file = self.config['stage1_output_file']
        # Verify that the stage1 output file exists
        if not os.path.exists(stage1_output_file):
            raise FileNotFoundError(f"Stage 1 output file {stage1_output_file} not found")

        print("Running Stage 2 Inference")
        for model in tqdm(self.config['stage_2_models']):
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


    def stage3(self):
        print("Running Stage 3 Inference")
        stage3_workflow.stage_3_pipeline(
            stage_2_output_paths = [stage2['final_json_results_file'] for stage2 in self.config['stage_2_models']],
            stage_3_model_paths=self.config['stage_3_models'],
            output_file=self.config['results_output_file']
        )

    def run(self, db_id, db_path, question, evidence):

        db_metadata = get_db_metadata(db_id, db_path)
        

if __name__ == "__main__":
    inference = Inference('inference_config.yaml')
    inference.stage1()
    # inference.stage2()
    # inference.stage3()

    # Get results 
    run_eval.eval(
        inference.config['results_output_file'],
        inference.config['results_output_with_eval']
    )
