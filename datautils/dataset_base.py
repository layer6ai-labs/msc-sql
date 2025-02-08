import random
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from utils import load_jsonl_file, load_json_file,save_jsonl_file
from constants import stage3_preamble, stage3_post_2_sql

import re


class DatasetBase:

    dataset_path: list[str] | str
    dataset_name: list[str] | str
    ds: TorchDataset | HFDataset

    def __init__(self, dataset_path: list[str] | str, dataset_name: list[str] | str, **kwargs):
        """
            dataset_path: list[str] | str -> is either the path to one jsonl file, or multiple paths to jsonl files one for each dataset (spider, bird)
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.kwargs = kwargs

    def load_dataset(self, **kwargs):
        pass


class ChatDataset(DatasetBase):

    def __init__(self, dataset_path: list[str] | str, dataset_name: list[str] | str, **kwargs):
        super().__init__(dataset_path, dataset_name, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        percent = kwargs.get('percent', None)
        subsample_by_difficulty = kwargs.get('subsample_by_difficulty', False)

        if isinstance(self.dataset_path, list):
            assert len(self.dataset_path) == len(self.dataset_name), "Give a name for each dataset path provided"
            ds = None
            for path, name in zip(self.dataset_path, self.dataset_name):
                if ds == None:
                    ds = load_single_dataset(path, name, percent, subsample_by_difficulty)
                else:
                    ds += load_single_dataset(path, name, percent, subsample_by_difficulty)
        else:
            ds = load_single_dataset(self.dataset_path, self.dataset_name, percent, subsample_by_difficulty)

        self.ds = HFDataset.from_pandas(pd.DataFrame(ds))

    def __len__(self):
        return len(self.ds)

class ChatDatasetStage3From2Files(DatasetBase):

    def __init__(self, dataset_path: list[str] | str, dataset_name: list[str] | str, **kwargs):

        # Verify that the dataset path is a list of 2
        assert isinstance(dataset_path, list), "The dataset path should be a list of 2 paths"
        assert len(dataset_path) == 2, "The dataset path should be a list of 2 paths"

        super().__init__(dataset_path, dataset_name, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        percent = kwargs.get('percent', None)
        required_idxs = kwargs.get('required_idxs', None)

        if isinstance(self.dataset_path, list):
            file1 = self.dataset_path[0]
            file2 = self.dataset_path[1]

            ds1 = self.load_stage2_results(file1)
            ds2 = self.load_stage2_results(file2)



            ds = self.merge_stage2_files(ds1, ds2)

            # If percent is not None, then subsample the dataset
            if percent and percent < 1.0:
                random.seed(42)
                ds = random.sample(ds, int(len(ds) * percent))

            ds_stage_3 = self.create_stage3_dataset(ds, required_idxs)

        print(f"Total dataset len: {len(ds_stage_3)}")

        # go through ds_stage_3 and remove any None
        ds_stage_3 = [b for b in ds_stage_3 if b is not None]

        self.ds = HFDataset.from_pandas(pd.DataFrame(ds_stage_3))

    def create_stage3_dataset(self, ds, required_idxs=None):
        if not required_idxs:
            required_idxs = []
        
        dataset = []
        for b in ds:

            if b['idx'] not in required_idxs and (b['sql_pred'][0] == b['sql_pred'][1]):
                continue
            else:
                dataset.append(add_stage3_sample(b, b['sql_pred'], results=b['result'], id=b['idx']))

        return dataset
    

    def load_stage2_results(self, ds_path):
        # load the file and return the results
        if ".jsonl" in ds_path:
            ds = load_jsonl_file(ds_path)
        else:
            ds = load_json_file(ds_path)

        return ds

    def merge_stage2_files(self, ds1, ds2):
        # Combine them based on the index and return the combined dataset
        ds2_map = {b['idx']: b for b in ds2}
        for d1 in ds1:
            if not d1['idx'] in ds2_map:
                print(f"Index {d1['idx']} not found in the first dataset, skipping")
                continue
            else:
                d1['sql_pred'] = [d1['sql_pred'], ds2_map[d1['idx']]['sql_pred']]
                d1['result'] = [d1['result'], ds2_map[d1['idx']]['result'] ]

        return ds1

    def __len__(self):
        return len(self.ds)


def load_single_dataset(ds_path, dataset_name, percent, subsample_by_difficulty):
    ds = load_jsonl_file(ds_path) 

    # remove unwanted cols
    ds = [{
        'idx': f"{dataset_name}_{b['idx']:05}",
        'db_id': b['db_id'],
        'db_path': b['db_info']['db_path'],
        'user': b['gen']['user'],
        'assistant': b['gen']['assistant'],
        'difficulty': b['difficulty'] if 'difficulty' in b else "default",
    } for b in ds]

    if percent and percent < 1.0:
        random.seed(42)
        if subsample_by_difficulty:
            ds = subsample_each_difficulty(ds, percent)
        else:
            ds = random.sample(ds, int(len(ds) * percent))
  
    return ds


def add_stage3_sample(b, predictions, results=None, id=None):

    prompt = ""

    if len(predictions) == 2:

        if results == None:
            prompt = stage3_preamble +  b['question_prompt'] +  stage3_post_2_sql(predictions[0], predictions[1], "", "")
        else:
            prompt = stage3_preamble +  b['question_prompt'] +  stage3_post_2_sql(predictions[0], predictions[1], results[0], results[1])
    else:
        
        print("Unknown number of predictions ", id)
        print("skipping")
        return None
    
    assert prompt != "", "Prompt is empty"

    sample = {'idx': b['idx'] if id == None else id,
            'db_id': b['db_id'],
            'db_path': b['db_path'],
            'user': prompt,                                                             
            'sql_stage2_results': results,
            'sql_pred': predictions,
            'difficulty': b['difficulty'] if 'difficulty' in b else None,
        }

    return sample

def subsample_each_difficulty(ds, percent):

    simple = [b for b in ds if b['difficulty'] == 'simple']
    moderate = [b for b in ds if b['difficulty'] == 'moderate']
    challenging = [b for b in ds if b['difficulty'] == 'challenging']

    simple_subset = random.sample(simple, int(len(simple) * percent))
    moderate_subset = random.sample(moderate, int(len(moderate) * percent))
    challenging_subset = random.sample(challenging, int(len(challenging) * percent))

    print(f"Sub-Sampled by Difficulty: Simple: {len(simple_subset)}, Moderate: {len(moderate_subset)}, Challenging: {len(challenging_subset)}")

    # Print all the IDS
    all_samples = simple_subset + moderate_subset + challenging_subset
    # Print ids = 
    ids = [b['idx'] for b in all_samples]
    print(f"Sub-Sampled by Difficulty: IDS: {ids}") 
    return simple_subset + moderate_subset + challenging_subset

def format_as_single_chat_template(system=None, user=None, assistant=None):
    chat_template = []
    if system:
        chat_template.append({"role": "system", "content": system})
    if user:
        chat_template.append({"role": "user", "content": user})
    if assistant:
        chat_template.append({"role": "assistant", "content": assistant})

    return chat_template
