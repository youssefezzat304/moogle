"""transformer.py — Vision Transformer-like encoder for lunar WAC imagery.

Originally copied from the WAC_2_WAC repo (custom_transformer.py) so this
repo is self-contained. This version adds a dedicated, learnable retrieval
token (similar to a CLS token) that is prepended to the patch token sequence
before the transformer layers. The retrieval token is new — it was not part
of the architecture used to train the existing WAC2WAC checkpoint, so it is
randomly initialised here and must be trained separately (see encoder.py).
"""

import torch
from torch.nn.modules import Module


class MyTransformer(Module):
    def __init__(self, patch_size=16, image_size=256, img_channels=1,
                 hidden_dim=768, nheads=8, num_layers=6):
        """A simple transformer encoder with a prepended retrieval token.

        Args:
            patch_size (int):     Pixels per patch (one side). Default 16.
            image_size (int):     Input image resolution. Default 256.
            img_channels (int):   Input channels (1 for WAC). Default 1.
            hidden_dim (int):     Transformer embedding dimension. Default 768.
            nheads (int):         Attention heads. Default 8.
            num_layers (int):     Transformer encoder layers. Default 6.
        """
        super(MyTransformer, self).__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.conv_proj = torch.nn.Conv2d(
            img_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable positional embeddings — one per patch (the retrieval
        # token has no spatial position, so it does not get one of these).
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )

        # Dedicated retrieval token, prepended to the patch sequence so the
        # self-attention layers can learn to aggregate one retrieval vector
        # per image. Shape (1, 1, hidden_dim) so it broadcasts across the
        # batch via .expand() below.
        self.retrieval = torch.nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size,
                      f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size,
                      f"Wrong image width! Expected {self.image_size} but got {w}!")

        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed

        # Prepend the retrieval token — one copy per item in the batch.
        retrieval_tokens = self.retrieval.expand(n, -1, -1)   # (B, 1, hidden_dim)
        x = torch.cat([retrieval_tokens, x], dim=1)           # (B, n_h*n_w + 1, hidden_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            (B, N + 1, hidden_dim) token sequence — index 0 is the retrieval
            token, indices 1..N are the patch tokens.
        """
        x = self._process_input(x)
        return self.encoder(x)