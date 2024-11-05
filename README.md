# MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation

Coming Soon.


Follow these steps:

* Process the dataset (produces a metadata json file): 
    ```sh
        python dataset_preprocess.py --dataset bird --db_prefix_path data/bird/dev_20240627/dev_databases --tables_json_path data/bird/dev_20240627/dev_tables.json --out_metadata_path data/bird/dev_20240627/dev_metadata.json
    ```

* Index the database

    ```sh
        mkdir -p output/db_index
        python index_db.py --metadata_path data/bird/dev_20240627/dev_metadata.json --save_path output/db_index/bird_dev_20240627
    ```

* Run inference
    ```sh
    CUDA_VISIBLE_DEVICES=0 python inference.py
    ```
