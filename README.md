# MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation

Coming Soon.


### Running the BIRD Benchmark

* Download the BIRD dev/test dataset https://bird-bench.github.io/ and extract contents to a directory. 

* Setup the Docker environment and mount the BIRD dataset directory and model checkpoints to the container.

    ```sh
    docker build -t <image-name> .
    docker run -v /path/to/bird_dataset:/bird_data -v /path/to/model_checkpoints:/checkpoints <image-name>
    ```

* Process the dataset (produces a metadata json file): 
    ```sh
        python dataset_preprocess.py --dataset bird --db_prefix_path /bird_data/dev_20240627/dev_databases --tables_json_path /bird_data/dev_20240627/dev_tables.json --out_metadata_path /bird_data/dev_20240627/dev_metadata.json
    ```

* Index the database

    ```sh
        mkdir -p output/db_index
        python index_db.py --metadata_path /bird_data/dev_20240627/dev_metadata.json --save_path output/db_index/bird_dev_20240627
    ```

* Run inference
    ```sh
    CUDA_VISIBLE_DEVICES=0 python inference.py
    ```
