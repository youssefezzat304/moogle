from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch


class NgramTokenizerWrapper:
    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 256,
    ):
        self.tokenizer_path = Path(tokenizer_path)
        self.max_length = max_length

        with open(self.tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["model"]["vocab"]

        # Special tokens
        self.retrieval_id = self.vocab["[RETRIEVAL]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.sos_id = self.vocab["[SOS]"]
        self.eos_id = self.vocab["[EOS]"]
        self.mask_id = self.vocab["[MASK]"]

        # Reverse lookup table
        self.inverse_vocab = {
            idx: token
            for token, idx in self.vocab.items()
        }

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenization should match how the tokenizer was trained.
        """

        text = text.lower().strip()

        tokens = []

        for word in text.split():

            # Character 3-grams
            n = 3

            if len(word) < n:
                tokens.append(word)
                continue

            for i in range(len(word) - n + 1):
                tokens.append(word[i:i + n])

        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        add_retrieval_token: bool = False,
    ):

        tokens = self.tokenize(text)

        ids = []

        if add_special_tokens:

            if add_retrieval_token:
                ids.append(self.retrieval_id)

            ids.append(self.sos_id)

        for token in tokens:
            ids.append(
                self.vocab.get(
                    token,
                    self.unk_id,
                )
            )

        if add_special_tokens:
            ids.append(self.eos_id)

        return SimpleNamespace(ids=ids)

    def encode_retrieval(self, text: str):
        return self.encode(
            text,
            add_retrieval_token=True,
        )

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
        skip_special_tokens: bool = True,
    ) -> str:

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [
            self.inverse_vocab.get(i, "[UNK]")
            for i in token_ids
        ]

        if skip_special_tokens:
            special = {
                "[RETRIEVAL]",
                "[PAD]",
                "[UNK]",
                "[SOS]",
                "[EOS]",
                "[MASK]",
            }

            tokens = [
                token
                for token in tokens
                if token not in special
            ]

        return " ".join(tokens)

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def id_to_token(self, idx: int) -> str | None:
        return self.inverse_vocab.get(idx)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)