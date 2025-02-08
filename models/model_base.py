import torch
from typing import Any, Callable
from peft.auto import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig, AutoTokenizer, PreTrainedTokenizerBase


class ModelAndTokenizer:
    model_name: str
    model: PreTrainedModel
    model_config: Any
    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int
    collator_prefix_str: str
    completion_str_fn: Callable[[str], str]
    formatting_function: Callable[[PreTrainedTokenizerBase], Callable]

    def __init__(self,
                 model_name_or_path,
                 quantization_config=None,
                 device_map="auto",
                 attn_implementation=None,
                 torch_dtype=torch.bfloat16,
                 training=True,
                 peft_model=False):
        self.model_name = model_name_or_path

        if not peft_model:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              device_map=device_map,
                                                              attn_implementation=None if not attn_implementation else attn_implementation,
                                                              torch_dtype=torch_dtype,
                                                              quantization_config=quantization_config)

            self.model_config = AutoConfig.from_pretrained(self.model_name)
        else:
            self.model = AutoPeftModelForCausalLM.from_pretrained(self.model_name,
                                                                  device_map=device_map,
                                                                  attn_implementation=None if not attn_implementation else attn_implementation,
                                                                  torch_dtype=torch.bfloat16,
                                                                  use_cache=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = 'left'
        self.formatting_function = conversations_formatting_function(self.tokenizer)

        if not training:
            self.model.eval()
            self.model = torch.compile(self.model)


class MistralModelAndTokenizer(ModelAndTokenizer):
    def __init__(self,
                 model_name_or_path,
                 quantization_config=None,
                 device_map="auto",
                 attn_implementation=None,
                 torch_dtype=torch.bfloat16,
                 training=True,
                 peft_model=False):
        super().__init__(model_name_or_path, quantization_config, device_map, attn_implementation, torch_dtype, training, peft_model)
        self.collator_prefix_str = "[/INST]"
        self.completion_str_fn = lambda prompt: f"<s>[INST] {prompt} [/INST]"
        self.max_seq_len = 12000

        # while training we don't want pad token to eos because (to avoid generating multiple eos tokens)
        # during generation we want pad token to be eos token to stop generation used in batch decode.
        if training:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class GemmaModelAndTokenizer(ModelAndTokenizer):
    def __init__(self,
                 model_name_or_path,
                 quantization_config=None,
                 device_map="auto",
                 attn_implementation=None,
                 torch_dtype=torch.bfloat16,
                 training=True,
                 peft_model=False):
        super().__init__(model_name_or_path, quantization_config, device_map, attn_implementation, torch_dtype, training, peft_model)
        self.collator_prefix_str = "<start_of_turn>model"
        self.completion_str_fn = lambda prompt: f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.max_seq_len = 8192


class LlamaModelAndTokenizer(ModelAndTokenizer):
    def __init__(self,
                 model_name_or_path,
                 quantization_config=None,
                 device_map="auto",
                 attn_implementation=None,
                 torch_dtype=torch.bfloat16,
                 training=True,
                 peft_model=False):
        super().__init__(model_name_or_path, quantization_config, device_map, attn_implementation, torch_dtype, training, peft_model)
        print('llama here')
        self.collator_prefix_str = "<|start_header_id|>assistant<|end_header_id|>"
        self.completion_str_fn = lambda prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        self.max_seq_len = 8192

        print('setting pad token = eos token')
        self.tokenizer.pad_token = self.tokenizer.eos_token


def conversations_formatting_function(tokenizer: PreTrainedTokenizerBase):
    r"""
    return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples['conversations'][0], list):
            output_texts = []
            for i in range(len(examples['conversations'])):
                output_texts.append(tokenizer.apply_chat_template(examples['conversations'][i], tokenize=False))
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples['conversations'], tokenize=False)

    return format_dataset
