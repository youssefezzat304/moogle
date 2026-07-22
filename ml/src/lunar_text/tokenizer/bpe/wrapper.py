import torch

from lunar_text.utils.tokenizers import (
    REQUIRED_SPECIAL_TOKENS,
    load_tokenizer,
    pad_token_ids,
    special_token_ids,
)


class BPETokenizerWrapper:
    def __init__(self, tokenizer_path: str, max_length: int = 256):
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_length = max_length
        token_ids = special_token_ids(self.tokenizer, REQUIRED_SPECIAL_TOKENS)
        self.pad_id = token_ids["[PAD]"]
        self.retrieval_id = token_ids["[RETRIEVAL]"]
        self.sos_id = token_ids["[SOS]"]
        self.eos_id = token_ids["[EOS]"]
        self.mask_id = token_ids["[MASK]"]
        self.unk_id = token_ids["[UNK]"]

    def encode(self, text: str, add_retrieval_token: bool = False) -> torch.Tensor:
        encoded = self.tokenizer.encode(text)
        prefix = [self.sos_id]
        if add_retrieval_token:
            prefix = [self.retrieval_id, *prefix]
        ids = prefix + encoded.ids + [self.eos_id]

        if len(ids) > self.max_length:
            ids = ids[:self.max_length - 1] + [self.eos_id]

        return pad_token_ids(ids, pad_token_id=self.pad_id, max_length=self.max_length)

    def encode_retrieval(self, text: str) -> torch.Tensor:
        return self.encode(text, add_retrieval_token=True)

    def decode(self, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
