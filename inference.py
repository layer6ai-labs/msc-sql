from tqdm import tqdm
from utils import load_json_file
from constants import DATASETS
from dataset_gen import generate_instructions_for_model, generate_instruction_data_schema_linking
from agents.base_index_agent import BaseBirdFromTablePredAgent, BaseSpiderFromTablePredAgent

import stage1_prediction
import stage2_generate
import stage3_workflow
import yaml
import os
import run_eval


class Inference:
    def __init__(self, config_file, input_file, db_metadata_file, db_index_path, dataset=DATASETS.BIRD_DATASET):
        # Load the YAML configuration
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.config = config
        self.db_metadata = load_json_file(db_metadata_file)
        self.db_index_path = db_index_path
        # this is either the dev.json or test.json with fields {qid, question, evidence, db_id}
        self.input_data = load_json_file(input_file)
        self.dataset = dataset

    def stage1(self):
        # generate stage1 input files from dev.json and metadata.json
        generate_instruction_data_schema_linking(
            dataset=self.dataset,
            input_data=self.input_data,
            db_metadata=self.db_metadata,
            output_file=self.config['stage1_input_file'],
        )
        
        print('Running stage1 pipeline')
        stage1_input_file = self.config['stage1_input_file']
        # Verify that the stage1 output file exists
        if not os.path.exists(stage1_input_file):
            raise FileNotFoundError(f"Stage 1 input file {stage1_input_file} not found")

        stage1_prediction.stage_1_pipeline(
            model_name="/checkpoints/mistral_7b_schema_linking", # "mistral_7b_schema_linking", # TODO make this an argument
            peft_model=True,
            eval_ds_path=stage1_input_file,
            eval_batch_size=2,
            dataset_name='bird',
            eval_percent=1,
            intermediate_jsonl_results_file=self.config['stage1_temp_file'],
            final_json_results_file=self.config['stage1_output_file']
        )
        
    def stage2(self):

        # Using the stage1 output file, run stage2 inference
        stage1_output_file = self.config['stage1_output_file']
        if self.dataset == DATASETS.BIRD_DATASET:
            agent = BaseBirdFromTablePredAgent(self.db_index_path, stage1_output_file)
        else:
            agent = BaseSpiderFromTablePredAgent(self.db_index_path, stage1_output_file)

        _ = generate_instructions_for_model(
            dataset=self.dataset,
            input_data=self.input_data,
            db_metadata=self.db_metadata,
            agent=agent,
            output_file=self.config['stage2_input_file']
        )

        stage2_input_file = self.config['stage2_input_file']
        # Verify that the stage1 output file exists
        if not os.path.exists(stage2_input_file):
            raise FileNotFoundError(f"Stage 1 output file {stage2_input_file} not found")

        print("Running Stage 2 Inference")
        for model in tqdm(self.config['stage_2_models']):
            stage2_generate.stage_2_pipeline(
                model_name=model['model_name'],
                peft_model=model['peft_model'],
                eval_ds_path=stage2_input_file,
                eval_batch_size=1,
                dataset_name='bird',
                eval_percent=1,
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


if __name__ == "__main__":
    inference = Inference('inference_config.yaml', 
                          input_file="/bird_data/bird/dev/dev.json",  # make this an argument
                          db_metadata_file="/bird_data/bird/dev/dev_metadata.json",  # make this an argument
                          db_index_path='output/db_index/bird_dev_20240627')    # make this an argument
    inference.stage1()
    inference.stage2()
    inference.stage3()

    # Get results 
    # run_eval.eval(
    #     inference.config['results_output_file'],
    #     inference.config['results_output_with_eval']
    # )
