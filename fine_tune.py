import torch
import argparse
from pprint import pprint
from peft import LoraConfig
from datasets import Dataset
from typing import Type, Literal
from config import config_factory
from config.ft_base import FTBase
from model_eval import do_eval
from models.model_base import ModelAndTokenizer
from models.model_factory import get_model_class
from torch.utils.tensorboard import SummaryWriter
from datautils.dataset_base import ChatDataset, DatasetBase
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, TrainingArguments



class FineTuner:
    
    train_ds: DatasetBase
    eval_ds: DatasetBase

    def __init__(self, model_and_tok: ModelAndTokenizer, train_ds: DatasetBase, eval_ds: DatasetBase, config: FTBase):
        self.model_and_tok: ModelAndTokenizer = model_and_tok
        self.collator = DataCollatorForCompletionOnlyLM(self.model_and_tok.collator_prefix_str, tokenizer=self.model_and_tok.tokenizer)
        self.train_ds = train_ds
        self.eval_ds = eval_ds

        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        # https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
        self.peft_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            r=config.lora_r,
            bias=config.lora_bias,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM", 
        )

        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            optim=config.optim,
            logging_steps=config.logging_steps,
            #save_strategy=config.save_strategy,
            learning_rate=config.learning_rate,
            bf16=config.bf16,                                        # only in Ampere architecture
            tf32=True if config.bf16 else False,                     # only in Ampere architecture
            fp16=True if not config.bf16 else False,
            max_grad_norm=config.max_grad_norm,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            push_to_hub=False,
            report_to=["tensorboard"],
            save_steps=100,
        )

        self.trainer = SFTTrainer(
            model=self.model_and_tok.model,
            args=self.training_args,
            train_dataset=self.train_ds.ds,
            peft_config=self.peft_config,
            max_seq_length=self.model_and_tok.max_seq_len,
            tokenizer=self.model_and_tok.tokenizer,
            packing=False,
            data_collator=self.collator,
            dataset_kwargs={
                "add_special_tokens": False,  # TODO: inspect, We template with special tokens in chat template
                "append_concat_token": False, # No need to add additional separator token
            },
            formatting_func=self.model_and_tok.formatting_function,
            #callbacks=[CustomTrainerCallback(self.model_and_tok, self.eval_ds, config)]
        )

    def train(self):
        self.trainer.train()

    def save(self):
        self.trainer.save_model()


class CustomTrainerCallback(TrainerCallback):
    def __init__(self, model_and_tok: ModelAndTokenizer, eval_ds: DatasetBase, config: FTBase):
        self.model_and_tok = model_and_tok
        self.eval_ds = eval_ds
        self.output_dir = config.output_dir
        self.eval_batch_size = config.per_device_eval_batch_size
        self.tb_writer = None

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, tokenizer, **kwargs):
        print('starting epoch end callback')
        model.eval()
        model = model.to(dtype=torch.bfloat16, device="cuda:0")
        with torch.no_grad():
            results = do_eval(self.model_and_tok, self.eval_ds, self.output_dir, self.eval_batch_size, model=model, tokenizer=tokenizer)
        if not self.tb_writer:
            self.tb_writer = SummaryWriter(args.logging_dir)
        self.tb_writer.add_scalar('eval_accuracy', results['accuracy'], state.global_step)
        model.train()


def main(args):

    config = config_factory(args.config_name)
    # these mandatory fields are set in config.
    config.model_name = args.model_name
    config.train_ds_path = args.train_ds_path
    config.test_ds_path = args.test_ds_path
    config.output_dir = args.output_dir
    config.eval_percent = args.eval_percent

    print('initializing fine tuning with following config')
    pprint(config)

    all_datasets = [
        'output/datasets/bird_train.jsonl',
        'output/datasets/spider_syn_train.jsonl',
    ]
    all_ds_names = [
        'bird',
        'spider_syn_train',
    ]

    train_ds = ChatDataset(all_datasets, all_ds_names)
    eval_ds = ChatDataset(config.test_ds_path, 'bird', percent=config.eval_percent, subsample_by_difficulty=True)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.bnb_load_in_4bit,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_and_tok_cls: Type[ModelAndTokenizer] = get_model_class(config.model_name)
    model_and_tok = model_and_tok_cls(config.model_name,
                                      quantization_config=bnb_config,
                                      attn_implementation=config.attn_implementation,
                                      torch_dtype=torch.bfloat16 if config.bf16 else torch.float16)

    finetuner = FineTuner(model_and_tok, train_ds, eval_ds, config)
    finetuner.train()
    finetuner.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config type name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the model (either one of openai models or self-hosted ex: defog/sqlcoder-7b-2)"
    )
    # parser.add_argument(
    #     "--train_ds_path",
    #     type=str,
    #     required=True,
    #     help="train dataset path to a jsonl file"
    # )
    parser.add_argument(
        "--test_ds_path",
        type=str,
        required=True,
        help="test dataset path to a jsonl file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory to save model"
    )
    parser.add_argument(
        "--eval_percent",
        type=float,
        default=1.0,
        help="Percentage of evaluation samples to use"
    )
    args = parser.parse_args()
    main(args)
