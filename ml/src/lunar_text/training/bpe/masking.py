import torch


class Masking:
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int,
        special_token_ids: list[int] | None = None,
        mask_probability: float = 0.15,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be greater than 0. Got {vocab_size}.")

        if not 0 <= mask_probability <= 1:
            raise ValueError(
                f"mask_probability must be between 0 and 1. Got {mask_probability}."
            )

        if special_token_ids is None or len(special_token_ids) == 0:
            raise ValueError("special_token_ids must not be None or empty.")

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = sorted(set(special_token_ids) | {pad_token_id, mask_token_id})
        self.mask_probability = mask_probability

        self.random_token_ids = [
            token_id
            for token_id in range(vocab_size)
            if token_id not in self.special_token_ids
        ]
        if not self.random_token_ids:
            raise ValueError(
                "Vocabulary must contain at least one non-special token "
                "for random MLM replacement."
            )

    def __call__(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mask_probability,
            device=input_ids.device,
        )

        for token_id in self.special_token_ids:
            special_token_mask = labels == token_id
            probability_matrix.masked_fill_(special_token_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        if not masked_indices.any():
            valid_positions = probability_matrix > 0
            if valid_positions.any():
                valid_indices = valid_positions.nonzero(as_tuple=False)
                selected = valid_indices[
                    torch.randint(
                        valid_indices.shape[0],
                        (1,),
                        device=input_ids.device,
                    )
                ][0]
                masked_indices[tuple(selected)] = True

        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 0.8, device=input_ids.device)
            ).bool()
            & masked_indices
        )
        masked_input_ids[indices_replaced] = self.mask_token_id

        indices_random = (
            torch.bernoulli(
                torch.full(labels.shape, 0.5, device=input_ids.device)
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        valid_random_token_ids = torch.tensor(
            self.random_token_ids,
            dtype=torch.long,
            device=input_ids.device,
        )
        random_indices = torch.randint(
            valid_random_token_ids.shape[0],
            labels.shape,
            dtype=torch.long,
            device=input_ids.device,
        )
        random_words = valid_random_token_ids[random_indices]
        masked_input_ids[indices_random] = random_words[indices_random]

        return masked_input_ids, labels
