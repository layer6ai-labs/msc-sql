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


### Breakdown of the Arguments:

#### **Stage 1**:
- `--stage1_input_file`: Points to the Stage 1 input file.
  - Example: `schema_linking/bird_dev_table_selection_0910.json`
  
- `--stage1_temp_file`: Points to the temporary file for Stage 1 intermediate results.
  - Example: `results/stage1/intermediate_report.jsonl`
  
- `--stage1_output_file`: Points to the Stage 1 output file.
  - Example: `schema_linking/stage1_output.json`
  
- `--peft_model`: Boolean flag indicating whether to use a PEFT (Prompt-enhanced Fine-Tuning) model in Stage 1.
  - Example: `true`

#### **Stage 2**:
- `--stage2_input_file`: Points to the input file for Stage 2.
  - Example: `output/stage2_rag_input.json`
  
- `--stage2_model_names`: List of Stage 2 model paths to use.
  - Example: `"/checkpoints/mistral_model_april21_merged" "/checkpoints/gemma9b_july25_med/" "/checkpoints/llama3_with_archer_july5_high_all_datasets"`
  
- `--stage2_peft_models`: List of boolean flags indicating whether each model is a PEFT model.
  - Example: `false true true`
  
- `--stage2_intermediate_files`: List of paths to the intermediate results files for each model in Stage 2.
  - Example: `"output_mistral_report.jsonl" "output_gemma_report.jsonl" "output_llama_report.jsonl"`
  
- `--stage2_final_files`: List of paths to the final output files for each model in Stage 2.
  - Example: `"output_mistral_report.json" "output_gemma_report.json" "output_llama_report.json"`

#### **Stage 3**:
- `--stage3_models`: List of Stage 3 model paths to use.
  - Example: `"/checkpoints/stage3_mistralAUG22_CUDA1_fts_with_result_inputs/checkpoint-2400" "/checkpoints/stage3_mistralAUG21_fts_with_result_inputs/checkpoint-2400" "/checkpoints/stage3_mistralAUG22_CUDA0_fts_with_result_inputs/checkpoint-2400"`
  
- `--stage3_peft_model`: Boolean flag indicating whether to use a PEFT model in Stage 3.
  - Example: `false`
  
- `--results_output_file`: Points to the file where the final Stage 3 results will be stored.
  - Example: `output_stage3.json`

#### Sample Run:

```sh
python inference_pipeline.py \
  --run_stage1 \
  --input_file "/bird_data/bird/dev/dev.json" \
  --db_metadata_file "/bird_data/bird/dev/dev_metadata.json" \
  --db_index_path "output/db_index/bird_dev_20240627" \
  --stage1_input_file "schema_linking/bird_dev_table_selection_0910.json" \
  --stage1_temp_file "results/stage1/intermediate_report.jsonl" \
  --stage1_output_file "schema_linking/stage1_output.json" \
  --peft_model true \
  --run_stage2 \
  --stage2_input_file "output/stage2_rag_input.json" \
  --stage2_model_names "/checkpoints/mistral_model_april21_merged" "/checkpoints/gemma9b_july25_med/" "/checkpoints/llama3_with_archer_july5_high_all_datasets" \
  --stage2_peft_models false true true \
  --stage2_intermediate_files "output_mistral_report.jsonl" "output_gemma_report.jsonl" "output_llama_report.jsonl" \
  --stage2_final_files "output_mistral_report.json" "output_gemma_report.json" "output_llama_report.json" \
  --run_stage3 \
  --stage3_models "/checkpoints/stage3_mistralAUG22_CUDA1_fts_with_result_inputs/checkpoint-2400" "/checkpoints/stage3_mistralAUG21_fts_with_result_inputs/checkpoint-2400" "/checkpoints/stage3_mistralAUG22_CUDA0_fts_with_result_inputs/checkpoint-2400" \
  --stage3_peft_model true \
  --results_output_file "output_stage3.json"
```