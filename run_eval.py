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


def do_eval(model_and_tok: ModelAndTokenizer, eval_ds: DatasetBase, output_dir: str, eval_batch_size: int, eval_sql_each_gen=True, model=None, tokenizer=None):
    eval_ds = eval_ds.ds.map(lambda item: {'user_prompt': model_and_tok.completion_str_fn(item['user'])})

    model = model_and_tok.model if not model else model
    tokenizer = model_and_tok.tokenizer if not tokenizer else tokenizer
    
    bird_eval = BirdEvaluator()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jsonl_report_file_name = os.path.join(output_dir, 'report.jsonl')
    json_report_file_name = os.path.join(output_dir, 'report.json')

    dataloader = DataLoader(eval_ds, batch_size=eval_batch_size)
    pred_sqls, gt_sqls, all_data = [], [], []

    for batch in tqdm(dataloader):
        batch_of_prompts = batch['user_prompt']
        
        batch_of_tokens = tokenizer(batch_of_prompts, return_tensors='pt', padding=True).to("cuda:0")   # since always training on 1 gpu and using CUDA_VISIBLE_DEVICES, ok for now; fix later
        with torch.inference_mode():
            generate_ids = model.generate(**batch_of_tokens, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1000)
        batch_of_answers = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        for i in range(len(batch_of_answers)):
            
            predicted_sql = extract_last_sql_block(batch_of_answers[i])
            if not predicted_sql:
                predicted_sql = ''

            gt_sql = batch['ground_truth'][i]
            db_path = batch['db_path'][i]
            user = batch['user'][i]
            gt_instr = batch['assistant'][i]
            idx = batch['idx'][i]
            if 'difficulty' in batch:
                difficulty = batch['difficulty'][i]
            else:
                difficulty = "default"

            answer = batch_of_answers[i]

            pred_sqls.append({'sql': predicted_sql, "db_path": db_path, "difficulty": difficulty, "idx": idx})
            gt_sqls.append({'sql': gt_sql, "db_path": db_path, "difficulty": difficulty, "idx": idx})
            
            all_data.append({
                'idx': idx,
                'sql_pred': predicted_sql,
                'sql_gt': gt_sql, 
                'question_prompt': user,
                'gen_answer': answer,
                'gt_answer': gt_instr,
                'db_id': batch['db_id'][i], 
                'db_path': db_path,
                'difficulty': difficulty
            })

            with open(jsonl_report_file_name, 'a') as f:
                f.write(json.dumps(all_data[-1], indent=4, sort_keys=True))
                f.write('\n')

    updated_data = []
    results = bird_eval.evaluate(pred_sqls, gt_sqls, analyze=True)

    # results['individual_results'] have the same order as pred_sqls and gt_sqls
    for i, data in enumerate(all_data):
        data['eval_result'] = results['individual_results'][i]
        data['eval_analysis'] = results['analysis']
        updated_data.append(data)

        # Update the file with the results that include the evaluation
        with open(jsonl_report_file_name, 'a+') as f:
            f.write(json.dumps(updated_data, indent=4, sort_keys=True))
            f.write('\n')

    with open(json_report_file_name, 'w') as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))

    print('Accuracy:', results['exec'])
    print('SQL analysis:', results['analysis'])
    print("Accuracy by difficulty:", results['accuracy_by_difficulty'])
    return {'accuracy': results['exec']}


def eval(input_file, output_file):
    # Load the input file 
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    pred_sqls, gt_sqls = [], []

    # Each of the entries in the input file should have the following
    # 'db_path, `sql_pred`, `sql_gt`, `difficulty`, `idx`
    for entry in input_data:
        db_path = entry['db_path']
        sql_pred = entry['sql_pred']
        sql_gt = entry['sql_gt']
        difficulty = entry['difficulty']
        idx = entry['idx']


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
    parser.add_argument(
        "--eval_output_path",
        type=str,
        required=True,
        help="Path to the output file"
    )
    args = parser.parse_args()
    eval(args.eval_input_path, args.eval_output_path)
