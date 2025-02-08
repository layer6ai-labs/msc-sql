from typing import Type
from models.model_base import GemmaModelAndTokenizer, MistralModelAndTokenizer, ModelAndTokenizer, LlamaModelAndTokenizer


def get_model_class(model_name) -> Type[ModelAndTokenizer]:
    match model_name:
        case model_name if "mistral" in model_name.lower():
            return MistralModelAndTokenizer
        case model_name if "gemma" in model_name.lower():
            return GemmaModelAndTokenizer
        case model_name if "llama" in model_name.lower():
            return LlamaModelAndTokenizer
        case _:
            raise NotImplementedError("No such model class")

