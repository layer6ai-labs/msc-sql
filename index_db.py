import argparse
from tqdm import tqdm
from utils import load_json_file
from models.embedding_base import DBEmbedder
from sentence_transformers import SentenceTransformer

def index_all_databases(embedder, db_metadata, save_path='output/db_index/bird_train/'):
    for db_id in tqdm(list(db_metadata.keys())):
        db_path = db_metadata[db_id]['db_path']
        embedder.index_db(db_id, db_path, db_metadata[db_id])
        embedder.save_db_index(save_path, db_id)
        embedder.delete_index()


def main(metadata_path, save_path):
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    embedder = DBEmbedder(model)
    db_metadata = load_json_file(metadata_path)
    index_all_databases(embedder, db_metadata, save_path) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    
    metadata_path = args.metadata_path
    save_path = args.save_path

    # possible values
    # metadata_path = 'data/bird/train/train_metadata.json' 
    # metadata_path = 'data/bird/dev_20240627/dev_metadata.json' 
    # metadata_path = 'data/spider/train_metadata.json' 
    # metadata_path = 'data/spider/test_data/test_metadata.json' 
    # metadata_path = 'data/archer/train_metadata.json' 
    # metadata_path = 'data/archer/dev_metadata.json' 
    # save_path = 'output/db_index/bird_train' 
    # save_path = 'output/db_index/bird_dev_20240627' 
    # save_path = 'output/db_index/spider_train' 
    # save_path = 'output/db_index/spider_test' 
    # save_path = 'output/db_index/spider_test_2k' 
    # save_path = 'output/db_index/archer_train' 
    # save_path = 'output/db_index/archer_dev' 

    print('dataset from: ', metadata_path)
    print('saving to: ', save_path)
    main(metadata_path, save_path)
