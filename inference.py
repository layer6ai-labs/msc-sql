import argparse
from tqdm import tqdm
import os
from utils import load_json_file
from constants import DATASETS
from dataset_gen import generate_instruction_data_schema_linking
from agents.base_index_agent import BaseBirdFromTablePredAgent, BaseSpiderFromTablePredAgent
import stage1_prediction
import stage2_generate
import stage3_workflow

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    elif value.lower() in {'false', '0', 'no'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Inference:
    def __init__(self, args):
        self.args = args

        # Load database metadata
        self.db_metadata = load_json_file(args.db_metadata_file) if args.db_metadata_file else None
        self.db_index_path = args.db_index_path
        self.dataset = args.dataset
        self.input_data = load_json_file(args.input_file) if args.input_file else None

    def stage1(self):
        # Generate stage1 input files from dev.json and metadata.json
        generate_instruction_data_schema_linking(
            dataset=self.dataset,
            input_data=self.input_data,
            db_metadata=self.db_metadata,
            output_file=self.args.stage1_input_file,
        )

        print('Running Stage 1 pipeline')
        if not os.path.exists(self.args.stage1_input_file):
            raise FileNotFoundError(f"Stage 1 input file {self.args.stage1_input_file} not found")

        stage1_prediction.stage_1_pipeline(
            model_name=self.args.stage1_model,
            peft_model=self.args.peft_model,
            eval_ds_path=self.args.stage1_input_file,
            eval_batch_size=self.args.eval_batch_size,
            dataset_name='bird',
            eval_percent=1,
            intermediate_jsonl_results_file=self.args.stage1_temp_file,
            final_json_results_file=self.args.stage1_output_file
        )

    def stage2(self):
        print("Running Stage 2 Inference")
        if not os.path.exists(self.args.stage2_input_file):
            raise FileNotFoundError(f"Stage 2 input file {self.args.stage2_input_file} not found")

        # Load Stage 2 models from command-line arguments
        for i, model_name in enumerate(self.args.stage2_model_names):
            peft_model = self.args.stage2_peft_models[i]
            intermediate_file = self.args.stage2_intermediate_files[i]
            final_file = self.args.stage2_final_files[i]

            stage2_generate.stage_2_pipeline(
                model_name=model_name,
                peft_model=peft_model,
                eval_ds_path=self.args.stage2_input_file,
                eval_batch_size=self.args.eval_batch_size,
                dataset_name='bird',
                eval_percent=1,
                intermediate_jsonl_results_file=intermediate_file,
                final_json_results_file=final_file
            )

    def stage3(self):
        print("Running Stage 3 Inference")
        stage3_workflow.stage_3_pipeline(
            stage_2_output_paths=self.args.stage2_final_files,
            stage_3_model_paths=self.args.stage3_models,
            peft_model=self.args.stage3_peft_model,
            output_file=self.args.results_output_file
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference Pipeline")

    # General Arguments
    parser.add_argument('--dataset', default=DATASETS.BIRD_DATASET, help='Dataset to use')
    parser.add_argument('--input_file', help='Path to input JSON file')
    parser.add_argument('--db_metadata_file', help='Path to DB metadata JSON file')
    parser.add_argument('--db_index_path', help='Path to DB index')

    # Stage 1 Arguments
    parser.add_argument('--run_stage1', action='store_true', help='Run Stage 1')
    parser.add_argument('--stage1_input_file', help='Stage 1 input file path')
    parser.add_argument('--stage1_temp_file', help='Stage 1 temp file path')
    parser.add_argument('--stage1_output_file', help='Stage 1 output file path')
    parser.add_argument('--stage1_model', default='/checkpoints/mistral_7b_schema_linking', help='Stage 1 model name')
    parser.add_argument('--peft_model', type=str2bool, default=True, help='Use PEFT model for Stage 1')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Evaluation batch size')

    # Stage 2 Arguments
    parser.add_argument('--run_stage2', action='store_true', help='Run Stage 2')
    parser.add_argument('--stage2_input_file', help='Stage 2 input file path')
    parser.add_argument('--stage2_model_names', nargs='+', help='List of Stage 2 model names')
    parser.add_argument('--stage2_peft_models', nargs='+', type=str2bool, help='List of booleans for Stage 2 PEFT models')
    parser.add_argument('--stage2_intermediate_files', nargs='+', help='List of Stage 2 intermediate JSONL files')
    parser.add_argument('--stage2_final_files', nargs='+', help='List of Stage 2 final JSON files')

    # Stage 3 Arguments
    parser.add_argument('--run_stage3', action='store_true', help='Run Stage 3')
    parser.add_argument('--stage3_models', nargs='+', help='List of Stage 3 model paths')
    parser.add_argument('--stage3_peft_model', type=str2bool, default=False, help='Use PEFT model for Stage 3')
    parser.add_argument('--results_output_file', help='Path to save final results')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inference = Inference(args)

    if args.run_stage1:
        inference.stage1()
    if args.run_stage2:
        inference.stage2()
    if args.run_stage3:
        inference.stage3()
