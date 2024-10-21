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


def eval(args):
    eval_ds = ChatDataset(args.eval_ds_path, 
                          args.dataset_name, 
                          percent=args.eval_percent, 
                          subsample_by_difficulty=args.subsample_by_difficulty)

    model_and_tok_cls = get_model_class(args.model_name)

    if args.peft_model:
        model_and_tok = model_and_tok_cls(args.model_name, peft_model=True, training=False)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_and_tok = model_and_tok_cls(args.model_name, quantization_config=bnb_config, training=False)
    return do_eval(model_and_tok, eval_ds, args.output_dir, args.eval_batch_size, args.eval_sql_each_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Config type name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="dir to store reports"
    )
    parser.add_argument(
        "--peft_model",
        action='store_true',
        help="is this a peft model to load as AutoPeftModelForCausalLM?"
    )
    parser.add_argument(
        "--eval_ds_path",
        type=str,
        required=True,
        help="test dataset path to a jsonl file",
    )
    parser.add_argument(
        "--report_name",
        type=str,
        required=True,
        help="path to output report",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--eval_sql_each_gen",
        action="store_true",
        help="Run the SQL evaluation for each generation instead of everything once at the end."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='bird',
        help="path to output report",
    )
    parser.add_argument(
        "--subsample_by_difficulty",
        action='store_true',
        help="Run the evaluation on a subsample of the dataset with equal difficulty distribution as the original dataset."
    )
    parser.add_argument(
        "--eval_percent",
        type=float,
        default=1.0,
        help="Percentage of evaluation samples to use"
    )
    args = parser.parse_args()
    print(args)
    with torch.no_grad():
        eval(args)
