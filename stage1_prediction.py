import os
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from datautils.dataset_base import ChatDataset, DatasetBase
from datautils.sql_results import process_results_entry

from models.model_base import ModelAndTokenizer
from models.model_factory import get_model_class
from utils import extract_table_name_list



def generate_stage_1(
    model_and_tok: ModelAndTokenizer,
    eval_ds: DatasetBase,
    eval_batch_size: int,
    model=None,
    tokenizer=None,
    temperature=1.0,
    pass_k=5,
    intermediate_jsonl_results_file=None,
    final_json_results_file=None,
):
    """
    Perform schema linking (table selection) on a given dataset using a model and tokenizer, and save the results into json files.
    Args:
        model_and_tok (ModelAndTokenizer): The model and tokenizer to use for generation.
        eval_ds (DatasetBase): The dataset to evaluate on.
        eval_batch_size (int): The batch size for evaluation.
        model (optional): The model to use for evaluation. If not provided, the model from `model_and_tok` will be used.
        tokenizer (optional): The tokenizer to use for generation. If not provided, the tokenizer from `model_and_tok` will be used.
        temperature (int, optional): The temperature for generating SQL queries. Defaults to 1.
        pass_k (int, optional): The number of samples needed for ensemble.
        intermediate_jsonl_results_file (str, optional): The file to save intermediate results to. Defaults to None.
        final_json_results_file (str, optional): The file to save final results to. Defaults to None.
    
    """

    eval_ds = eval_ds.ds.map(
        lambda item: {"user_prompt": model_and_tok.completion_str_fn(item["user"])}
    )

    model = model_and_tok.model if not model else model
    tokenizer = model_and_tok.tokenizer if not tokenizer else tokenizer
    model = model.to(dtype=torch.bfloat16, device="cuda:0")

    dataloader = DataLoader(eval_ds, batch_size=eval_batch_size)
    all_data = []

    for batch in tqdm(dataloader):
        batch_of_prompts = batch["user_prompt"]
        batch_of_tokens = tokenizer(
            batch_of_prompts, return_tensors="pt", padding=True
        ).to(
            "cuda:0"
        )  # since always training on 1 gpu and using CUDA_VISIBLE_DEVICES, ok for now; fix later
        with torch.inference_mode():
            generate_kwargs = {
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": 1000,
            }

            if temperature != 0:
                generate_kwargs.update(
                    {
                        "temperature": temperature,
                        "do_sample": True,
                        "num_return_sequences": pass_k,
                    }
                )

            generate_ids = model.generate(**batch_of_tokens, **generate_kwargs)

        # single sampling
        # batch_of_answers = tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True
        # )

        # multi-sampling
        for idx_out in range(eval_batch_size):
            # samples for a given query
            batch_of_answers = []
            for idx_rep in range(pass_k):  
                batch_of_answers.append(
                    tokenizer.decode(
                        generate_ids[idx_out*pass_k + idx_rep][len(batch_of_tokens[idx_out]):], skip_special_tokens=True
                    )
                )
            
            predicted_tables_list = []
            predicted_tables_union = []
            for i in range(pass_k):
                predicted_tables = extract_table_name_list(batch_of_answers[i])
                if not predicted_tables or type(predicted_tables) != list:
                    predicted_tables = []
                predicted_tables_list.append(predicted_tables)
                predicted_tables_union += predicted_tables
            
            # get union 
            try:
                predicted_tables_union = list(set(predicted_tables_union))
            except:
                predicted_tables_union = []

            db_path = batch["db_path"][idx_out]
            user = batch["user"][idx_out]
            gt_instr = batch["assistant"][idx_out]
            idx = batch["idx"][idx_out]
            if "difficulty" in batch:
                difficulty = batch["difficulty"][idx_out]
            else:
                difficulty = "default"

            if "ground_truth" in batch:
                gt_tables = batch["ground_truth"][idx_out]
            else:
                gt_tables = []

            answer = batch_of_answers

            all_data.append(
                {
                    "idx": idx,
                    "tables_pred_diverse": predicted_tables_list,
                    "tables_pred_union": predicted_tables_union,
                    "tables_gt": gt_tables, 
                    "question_prompt": user,
                    "gen_answer": answer,
                    "gt_answer": gt_instr,
                    "db_id": batch["db_id"][idx_out],
                    "db_path": db_path,
                    "difficulty": difficulty,
                }
            )

            if intermediate_jsonl_results_file:
                with open(intermediate_jsonl_results_file, "a") as f:
                    f.write(json.dumps(all_data[-1], indent=4, sort_keys=True))
                    f.write("\n")

    with open(final_json_results_file, "w") as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))

def stage_1_pipeline(
    model_name,
    peft_model,
    eval_ds_path,
    eval_batch_size=1,
    dataset_name='bird',
    subsample_by_difficulty=True,
    eval_percent=1.0,
    temperature=1.0,
    pass_k=5,
    intermediate_jsonl_results_file='report.jsonl',
    final_json_results_file='final_report.json',
):
    """
        Executes the stage 1 pipeline for model evaluation.
    Args:
        model_name (str): The name of the model.
        peft_model (bool): Whether to use the PEFT model.
        eval_ds_path (str): The path to the evaluation dataset.
        eval_batch_size (int): The batch size for evaluation.
        dataset_name (str): The name of the dataset.
        subsample_by_difficulty (bool): Whether to subsample the dataset by difficulty.
        eval_percent (float): The percentage of the dataset to evaluate.
        temperature (float): The temperature for sampling.
        intermediate_jsonl_results_file (str): The path to save intermediate JSONL results.
        final_json_results_file (str): The path to save final JSON results.
    Returns:
        str: The path to the final JSON results file.
    """

    eval_ds = ChatDataset(
        eval_ds_path,
        dataset_name,
        percent=eval_percent,
        subsample_by_difficulty=subsample_by_difficulty,
    )
    with torch.no_grad():
        model_and_tok_cls = get_model_class(model_name)
        if peft_model:
            model_and_tok = model_and_tok_cls(
                model_name, peft_model=True, training=False
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_and_tok = model_and_tok_cls(
                model_name, quantization_config=bnb_config, training=False
            )

        return generate_stage_1(
            model_and_tok,
            eval_ds,
            eval_batch_size,
            temperature=temperature,
            pass_k=pass_k,
            intermediate_jsonl_results_file=intermediate_jsonl_results_file,
            final_json_results_file=final_json_results_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Config type name"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="dir to store reports"
    )
    parser.add_argument(
        "--peft_model",
        action="store_true",
        help="is this a peft model to load as AutoPeftModelForCausalLM?",
    )
    parser.add_argument(
        "--eval_ds_path",
        type=str,
        required=True,
        help="test dataset path to a jsonl file",
    )
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bird",
        help="path to output report",
    )
    parser.add_argument(
        "--subsample_by_difficulty",
        action="store_true",
        help="Run the evaluation on a subsample of the dataset with equal difficulty distribution as the original dataset.",
    )
    parser.add_argument(
        "--eval_percent",
        type=float,
        default=1.0,
        help="Percentage of evaluation samples to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling from the model. Default is 1.",
    )
    parser.add_argument(
        "--pass_k",
        type=int,
        default=5,
        help="Temperature for sampling from the model. Default is 5",
    )

    args = parser.parse_args()
    print(args)


    stage_1_pipeline(
        args.model_name,
        args.peft_model,
        args.eval_ds_path,
        args.eval_batch_size,
        args.dataset_name,
        args.subsample_by_difficulty,
        args.eval_percent,
        args.temperature,
        args.pass_k,
        os.path.join(args.output_dir, "intermediate_results.jsonl"),
        os.path.join(args.output_dir, "final_results.json"),
    )
