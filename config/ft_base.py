from dataclasses import dataclass
from typing import Literal

@dataclass
class FTBase:
    model_name: str
    train_ds_path: str
    test_ds_path: str

    attn_implementation: str

    # training args
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optim: str
    logging_steps: int
    save_strategy: str
    learning_rate: float
    bf16: bool
    fp16: bool
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str

    # evaluation args
    eval_percent: float
    per_device_eval_batch_size: int
    evaluation_strategy: str
    eval_steps: int

    # lora args
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: Literal['none', 'all', 'lora_only']
    lora_target_modules: str | list[str]

    # bitsandbytes args
    bnb_load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str


@dataclass
class SimpleConfig(FTBase):
    model_name: str = ''
    train_ds_path: str = ''
    test_ds_path: str = ''

    attn_implementation: str = ''

    # training args
    output_dir: str = ''
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 3
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    learning_rate: float = 0.0002
    bf16: bool = False
    fp16: bool = True
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'constant'

    # evaluation args
    eval_percent: float = 1.0
    per_device_eval_batch_size: int = 8
    evaluation_strategy: str = "epoch"
    eval_steps: int = 1

    # lora args
    lora_r: int = 32
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all', 'lora_only'] = "none"
    lora_target_modules: str | list[str] = "all-linear"

    # bitsandbytes args
    bnb_load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class AmpereConfig(FTBase):
    model_name: str = ''
    train_ds_path: str = ''
    test_ds_path: str = ''

    attn_implementation: str = 'flash_attention_2'

    # training args
    output_dir: str = ''
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 6
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch_fused"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    learning_rate: float = 0.00005
    bf16: bool = True
    fp16: bool = False
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = 'cosine'

    # evaluation args
    eval_percent: float = 1.0
    per_device_eval_batch_size: int = 2
    evaluation_strategy: str = "epoch"
    eval_steps: int = 1

    # lora args
    lora_r: int = 64 
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all', 'lora_only'] = "none"
    lora_target_modules: str | list[str] = "all-linear"

    # bitsandbytes args
    bnb_load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
