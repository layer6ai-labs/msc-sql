import os
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from datautils.dataset_base import ChatDataset, DatasetBase
from metrics.bird.evaluator import BirdEvaluator

from models.model_base import ModelAndTokenizer
from models.model_factory import get_model_class
from utils import extract_last_sql_block



def eval(input_file, gt_file, output_file):
    # Load the input file 
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # Load the ground truth data
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)


    gt_lookup = {entry['idx']: entry['sql_gt'] for entry in gt_data}

    pred_sqls, gt_sqls = [], []

    # Each of the entries in the input file should have the following
    # 'db_path, `sql_pred`, `sql_gt`, `difficulty`, `idx`
    for entry in tqdm(input_data):
        db_path = entry['db_path']
        sql_pred = entry['sql_pred']
        difficulty = entry['difficulty']
        idx = entry['idx']

        # Get the ground truth SQL
        sql_gt = gt_lookup[idx]


        # Evaluate the SQL
        pred_sqls.append({'sql': sql_pred, "db_path": db_path, "difficulty": difficulty, "idx": idx})
        gt_sqls.append({'sql': sql_gt, "db_path": db_path, "difficulty": difficulty, "idx": idx})

    bird_eval = BirdEvaluator()
    results = bird_eval.evaluate(pred_sqls, gt_sqls, analyze=True)

    # results['individual_results'] have the same order as pred_sqls and gt_sqls
    output_data = []
    for i, data in enumerate(input_data):
        data['eval_result'] = results['individual_results'][i]
        data['eval_analysis'] = results['analysis']
        output_data.append(data)
    with open(output_file, 'w') as f:
        f.write(json.dumps(output_data, indent=4, sort_keys=True))


    print('Accuracy:', results['exec'])
    print('SQL analysis:', results['analysis'])
    print("Accuracy by difficulty:", results['accuracy_by_difficulty'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_input_path",
        type=str,
        required=True,
        help="Path to the input file"
    )
    # ground truth data 
    parser.add_argument(
        "--gt_input_path",
        type=str,
        required=True,
        help="Path to the ground truth file"
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        required=True,
        help="Path to the output file"
    )
    args = parser.parse_args()
    eval(args.eval_input_path, args.gt_input_path , args.eval_output_path)
