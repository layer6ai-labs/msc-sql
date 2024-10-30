import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from datautils.dataset_base import ChatDatasetStage3From2Files

from models.model_factory import get_model_class
from utils import extract_last_sql_block, load_json_file
import itertools
import random

class PredictionResult:
    def __init__(
        self,
        idx,
        model_name,
        stage2_file1,
        stage2_file2,
        stage2_predicted_value,
        stage3_predicted_value,
        difficulty,
        db_path,
        db_id,
    ):
        self.idx = idx
        self.model_name = model_name
        self.stage2_file1 = stage2_file1
        self.stage2_file2 = stage2_file2
        self.stage2_predicted_value = stage2_predicted_value
        self.stage3_predicted_value = stage3_predicted_value
        self.difficulty = difficulty
        self.db_path = db_path
        self.db_id = db_id

    def __str__(self):
        return f"""
        idx: {self.idx}
        model_name: {self.model_name}
        stage2_file1: {self.stage2_file1}
        stage2_file2: {self.stage2_file2}
        stage2_predicted_value: {self.stage2_predicted_value}
        stage3_predicted_value: {self.stage3_predicted_value}
        difficulty: {self.difficulty}
        db_path: {self.db_path}
        db_id: {self.db_id}
        """

class Stage3AgentManager:
    def __init__(self):
        self.predictions = {}

    def add_or_update_prediction(
        self,
        idx,
        model_name,
        stage2_file1,
        stage2_file2,
        stage2_predicted_value,
        stage3_predicted_value,
        difficulty,
        db_path,
        db_id,
    ):

        if idx not in self.predictions:
            self.predictions[idx] = []
        
        self.predictions[idx].append(
            PredictionResult(
                idx,
                model_name,
                stage2_file1,
                stage2_file2,
                stage2_predicted_value,
                stage3_predicted_value,
                difficulty,
                db_path,
                db_id,
            )
        )

    def get_results_file(self):
        # go through each idx and write the results to a file only including the top voted model
        # For each idx, get the sql_pred of the model with the most votes
        # and keep all idx, stage3_predicted_value, difficulty, db_path
        # Correctness is irrelevant here
        results = []
        for idx, predictions in self.predictions.items():
            votes_per_stage2_options = {}
            for prediction in predictions:
                print("stage 3: ", prediction.stage2_predicted_value)
                if prediction.stage2_predicted_value not in votes_per_stage2_options:
                    votes_per_stage2_options[prediction.stage2_predicted_value] = 1
                else:
                    votes_per_stage2_options[prediction.stage2_predicted_value] += 1

            # get the model with the most votes
            max_votes = 0
            max_votes_stage2_option = None
            for model, votes in votes_per_stage2_options.items():
                if votes > max_votes:
                    max_votes = votes
                    max_votes_stage2_option = model

            results.append(
                {
                    "idx": idx,
                    "sql_pred": max_votes_stage2_option,
                    "difficulty": predictions[0].difficulty,
                    "db_path": predictions[0].db_path,
                }
            )

        return results


def eval_two_files(

    output_file,
    eval_ds_path_1,
    eval_ds_path_2,
    dataset_name,
    model_and_tok,
    temperature=0.0,
    required_idxs=None,
    eval_batch_size=1,
):
    """
    Evaluate stage 3 for two stage 2 files and generate a JSON report.
    Args:
        output_file (str): The path to the output JSON report file.
        eval_ds_path_1 (str): The path to the first evaluation dataset.
        eval_ds_path_2 (str): The path to the second evaluation dataset.
        dataset_name (str): The name of the dataset.
        model_and_tok (object): An object containing the model and tokenizer.
        temperature (float, optional): The temperature value for sampling. Defaults to 0.0.
        required_idxs (list, optional): A list of required indices. Defaults to None.
        eval_batch_size (int, optional): The batch size for evaluation. Defaults to 1.
    Returns:
        list: A list containing the evaluation data.
    """

    # Create two chat datasets for the two different orders
    eval_ds_v1 = ChatDatasetStage3From2Files(
        [eval_ds_path_1, eval_ds_path_2],
        dataset_name=dataset_name,
        percent=1.0,
        subsample_by_difficulty=True,
        stage2_eval=True,
        required_idxs=required_idxs,
        eval_batch_size=1,
        temperature=0.0,
    )

    eval_ds = eval_ds_v1.ds.map(
        lambda item: {"user_prompt": model_and_tok.completion_str_fn(item["user"])}
    )

    model = model_and_tok.model  # if not model else model
    tokenizer = model_and_tok.tokenizer  # if not tokenizer else tokenizer

    json_report_file_name = output_file

    dataloader = DataLoader(eval_ds, batch_size=eval_batch_size)
    all_data = []


    for batch in tqdm(dataloader):
        batch_of_prompts = batch["user_prompt"]

        batch_of_tokens = tokenizer(
            batch_of_prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            generate_ids = model.generate(
                **batch_of_tokens,
                max_length=12000,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        batch_of_answers = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True
        )

        for i in range(len(batch_of_answers)):

            predicted_stage3 = extract_last_sql_block(batch_of_answers[i])
            if not predicted_stage3:
                print("Output is incorectly formatted. Choosing random stage 3")
                predicted_stage3 = random.choice(["1", "2"])

            idx = batch["idx"][i]
            db_path = batch["db_path"][i]
            user = batch["user"][i]
            if "difficulty" in batch:
                difficulty = batch["difficulty"][i]
            else:
                difficulty = "default"

            answer = batch_of_answers[i]

            if eval_batch_size == 1:
                stage2_chosen = batch["sql_pred"][int(predicted_stage3)-1][0]
            else:
                stage2_chosen = batch["sql_pred"][int(predicted_stage3)-1][i]            

            all_data.append(
                {
                    "idx": idx,
                    "sql_pred": stage2_chosen, 
                    "stage3_pred": predicted_stage3,
                    "question_prompt": user,
                    "gen_answer": answer,
                    "db_id": batch["db_id"][i],
                    "db_path": db_path,
                    "difficulty": difficulty,
                }
            )

    with open(json_report_file_name, "w") as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))

    return all_data


def eval_all_files_one_model(

    model_name,
    prediction_files,
    peft_model=True,
    stage3AgentManager=None,
    required_idxs=None,
):
    """
    Evaluate all stage 2 output files for a single stage 3 model.
    Args:
        model_name (str): The name of the model.
        prediction_files (list): A list of prediction files.
        peft_model (bool, optional): Whether to use the PEFT model. Defaults to True.
        stage3AgentManager (object, optional): The stage3AgentManager object. Defaults to None.
        required_idxs (list, optional): A list of required indexes. Defaults to None.
    """

    # merged_peft_model =load_merged_models(args)
    model_and_tok_cls = get_model_class(model_name)

    if peft_model:
        model_and_tok = model_and_tok_cls(model_name, peft_model=True, training=False)
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

    model_and_tok.tokenizer.pad_token = model_and_tok.tokenizer.eos_token

    # Generate all possible unordered pairs
    prediction_file_pairs = list(itertools.combinations(prediction_files, 2))
    
    # Print the pairs
    print(
        "Evaluating stage 3 pairs for model: ",
        model_name,
        " with ",
        len(prediction_file_pairs),
        " pairs",
    )

    for pair in tqdm(
        prediction_file_pairs, desc="Evaluating pairs"
    ):
        file1_name = pair[0].replace("/", "-").split(".json")[0]
        file2_name = pair[1].replace("/", "-").split(".json")[0]
        model_name = model_name.replace("/", "-")

        # if the stage3_results folder does not exist, create it
        if not os.path.exists("stage3_results"):
            os.makedirs("stage3_results")

        if required_idxs:
            file_name = f"stage3_results/{model_name}_{file1_name}_{file2_name}_{len(required_idxs)}.json"
        else:
            file_name = f"stage3_results/{model_name}_{file1_name}_{file2_name}.json"

        # if the report file exists, load it and return the results
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                all_data = json.load(f)
            print("Loaded the existing report file. Previously evaluated.")
            pair_predictions = all_data
        else:
            pair_predictions = eval_two_files(
                file_name,
                pair[0],
                pair[1],
                "bird",
                model_and_tok,
                required_idxs=required_idxs,
            )

        if stage3AgentManager:
            for predictions in pair_predictions:
                stage3AgentManager.add_or_update_prediction(
                    predictions["idx"],
                    model_name,
                    pair[0],
                    pair[1],
                    predictions["sql_pred"],
                    predictions["stage3_pred"],
                    predictions["difficulty"],
                    predictions["db_path"],
                    predictions["db_id"],
                )

def stage_3_pipeline(
        stage_2_output_paths,
        stage_3_model_paths,
        output_file,
):
    """
    Executes the stage 3 pipeline for evaluating models.
    Args:
        stage2_output_paths (list): List of file paths for stage 2 output.
        stage3_model_paths (list): List of file paths for stage 3 models.
        output_file (str): File path to write the results.
    Returns:
        None
    """

    stage3AgentManager = Stage3AgentManager()

    # Load all the stage 2 files and find the idx where the sql_pred are different between the models, 
    # at least one of them is correct and one is wrong
    # Load the JSON files
    stage2_res = {}
    stage2_full_res = {}
    for file in stage_2_output_paths:
        res = load_json_file(file)
        for record in res:
            idx = record["idx"]

            # Add the record to the full results but only keep db_path, difficulty, and sql_pred, idx, db_id
            if idx not in stage2_full_res:
                stage2_full_res[idx] = {}
                stage2_full_res[idx]["db_path"] = record["db_path"]
                stage2_full_res[idx]["difficulty"] = record["difficulty"]
                stage2_full_res[idx]["sql_pred"] = record["sql_pred"]
                stage2_full_res[idx]["idx"] = record["idx"]
                stage2_full_res[idx]["db_id"] = record["db_id"]
                stage2_full_res[idx]["source"] = "stage2"

            if idx not in stage2_res:
                stage2_res[idx] = [record["sql_pred"]]
            else:
                stage2_res[idx].append(record["sql_pred"])



    # narrow down the idxs where the results are different between the models
    idxs_for_stage3 = []
    res_without_stage3 = []
    for idx, results in stage2_res.items():
        if len(set(results)) > 1:
            idxs_for_stage3.append(idx)
        else:
            res_without_stage3.append(stage2_full_res[idx])

    print(
        "Number of idxs where the results are different between the models: ", len(idxs_for_stage3)
    )

    for model in tqdm(stage_3_model_paths, desc="Evaluating models"):
        print("Evaluating model: ", model)
        with torch.no_grad():
            eval_all_files_one_model(
                model,
                stage_2_output_paths,
                peft_model=True,
                stage3AgentManager=stage3AgentManager,
                required_idxs=idxs_for_stage3,
            )


    results = stage3AgentManager.get_results_file()
    # Add back the results where the models agreed
    results.extend(res_without_stage3)
    # Write the results to a file
    with open(output_file, "w") as f:
        f.write(json.dumps(results, indent=4, sort_keys=True))


