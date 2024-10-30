import os
import json
import argparse
import multiprocessing
from tqdm import tqdm
from agents.base_random_agent import BaseBirdAgent, BaseSpiderAgent
from constants import DATASETS
from utils import load_json_file, load_jsonl_file


def init_bird_message(sql_sample: dict, db_metadata: dict) -> dict:                                                              
  """                                                                                                                          
  Construct message for text-to-SQL task given the Bird dataset                                                                
  :param sql_sample: one sample of dataset                                                                                     
  :param db_metadata: db metadata constructed by dataset_preprocess.py                                                         
  :return: initial message object of group chat                                                                                
  """                                                                                                                          
  idx, db_id, query, evidence, gt, difficulty = (                                                                              
      sql_sample["question_id"],                                                                                               
      sql_sample["db_id"],                                                                                                     
      sql_sample["question"],                                                                                                  
      sql_sample["evidence"],                                                                                                  
      sql_sample.get("SQL", ""),                                                                                               
      sql_sample.get("difficulty", "simple"),                                                                                  
  )                                                                                                                            
                                                                                                                               
  db_info = db_metadata[db_id]                                                                                                 
                                                                                                                               
  user_message = {                                                                                                             
      "idx": idx,                                                                                                              
      "db_id": db_id,                                                                                                          
      "db_info": db_info,                                                                                                      
      "query": query,                                                                                                          
      "evidence": evidence,                                                                                                    
      "ground_truth": gt,                                                                                                      
      "difficulty": difficulty,                                                                                                
  }                                                                                                                            
  return user_message                                                                                                          


def init_spider_message(sql_sample: dict, db_metadata: dict) -> dict:                                                            
  """                                                                                                                          
  Construct message for text-to-SQL task given the spider dataset                                                              
  :param sql_sample: one sample of dataset                                                                                     
  :param db_metadata: db metadata constructed by dataset_preprocess.py                                                         
  :return: initial message object of group chat                                                                                
  """                                                                                                                          
  idx, db_id, query, gt = (                                                                                                    
      sql_sample["question_id"],                                                                                               
      sql_sample["db_id"],                                                                                                     
      sql_sample["question"],                                                                                                  
      sql_sample.get("query", ""),                                                                                             
  )                                                                                                                            
                                                                                                                               
  db_info = db_metadata[db_id]                                                                                                 
  user_message = {                                                                                                             
      "idx": idx,                                                                                                              
      "db_id": db_id,                                                                                                          
      "db_info": db_info,                                                                                                      
      "query": query,                                                                                                          
      "ground_truth": gt,                                                                                                      
  }                                                                                                                            
  return user_message                                                                                                          


def get_agent(dataset, index_path=''):
    if dataset == DATASETS.BIRD_DATASET:
        agent = BaseBirdAgent(index_path)
    else:
        agent = BaseSpiderAgent(index_path)
    return agent


def construct_dataset(dataset, input_data, db_metadata):

    all_data_items = []
    for cur_idx, data_item in enumerate(tqdm(input_data)):

        if "question_id" not in data_item:
            data_item["question_id"] = cur_idx

        if dataset == DATASETS.BIRD_DATASET:
            data_item = init_bird_message(data_item, db_metadata)
        elif dataset == DATASETS.SPIDER_DATASET:
            data_item = init_spider_message(data_item, db_metadata)
        all_data_items.append(data_item)

    return all_data_items


def write_to_file(output_file, output_queue):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a+", encoding="utf-8") as fp:
        while True:
            data_item = output_queue.get()
            if data_item is None:
                break
            print(json.dumps(data_item, ensure_ascii=False), file=fp, flush=True)


def generate_instructions_for_model(dataset, input_data, db_metadata, agent, output_file):
    dataset_items = construct_dataset(dataset, input_data, db_metadata)
    with open(output_file, "a+", encoding="utf-8") as fp:
        for _, data_item in enumerate(tqdm(dataset_items)):
            agent_result = agent.talk(data_item)
            data_item['gen'] = agent_result
            print(json.dumps(data_item, ensure_ascii=False), file=fp, flush=True)
    return dataset_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the model (either one of openai models or self-hosted ex: defog/sqlcoder-7b-2)"
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="for openai models, leave it as an empty string, for others specify the hosted server url"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[c.value for c in DATASETS],
        help="The dataset to process(ex: bird, spider)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="path to dataset input (dev.json)"
    )
    parser.add_argument(
        "--db_metadata_file",
        type=str,
        required=True,
        help="path to preprocessed metadata file (see dataset_preprocess.py)",
    )
    parser.add_argument(
        "--db_index_path",
        type=str,
        default='',
        help="database index path used by the embedding model for nn lookup",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="output file path to save generated data at",
    )
    args = parser.parse_args()
    args.dataset = DATASETS(args.dataset)

    agent = get_agent(args.dataset, args.db_index_path)

    print('Using agent:', agent)
    print('Arguments')
    print(args)

    input_data = load_json_file(args.input_file)
    db_metadata = load_json_file(args.db_metadata_file)

    generate_instruction_data_sequentially(
        dataset=args.dataset,
        output_file=args.output_file,
        input_data=input_data,
        db_metadata=db_metadata,
        agent=agent,
    )
