import torchvision.transforms as transforms
import torch
from transformers import BertTokenizer


class TokenizeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        return torch.squeeze(torch.stack((
            tokens["input_ids"], tokens["attention_mask"], 
            tokens["token_type_ids"]), dim=2), dim=0)


class BertTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(
                tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))
        del train
