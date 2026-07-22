import torch
import torch.nn as nn

from lunar_vision.model.clip_backend import resize_pos_embed, source_pos_grid_size


# ──────────────────────────────────────────────
# ENCODER
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    """
    ViT-style patch encoder.
 
    Steps:
      1. Splits the image into non-overlapping patches via a strided Conv2d.
      2. Adds learnable positional embeddings.
      3. Passes the patch sequence through a Transformer Encoder.
 
    Args:
        patch_size  (int): Side length of each square patch.   Default 16.
        image_size  (int): Assumed square input resolution.    Default 256.
        img_channels(int): Number of input channels (e.g. 3). Default 3.
        hidden_dim  (int): Token / embedding dimension.        Default 512.
        nheads      (int): Attention heads per layer.          Default 8.
        num_layers  (int): Number of TransformerEncoderLayers. Default 6.
    """
 
    def __init__(
        self,
        patch_size: int = 16,
        image_size: int = 256,
        img_channels: int = 3,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
 
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        assert hidden_dim % nheads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by nheads ({nheads})"
        )
 
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2  # e.g. 16*16 = 256
 
        # Patch embedding: each patch → 1 token of size hidden_dim
        self.conv_proj = nn.Conv2d(
            img_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
 
        # Learnable positional embedding (one per patch position)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Dedicated retrieval token (similar to a CLS token), prepended to the
        # patch sequence only by encode_retrieval() below — never used by
        # forward()/decode(), so the reconstruction/autoencoder path this
        # class was originally trained for stays byte-for-byte unchanged.
        # This token is new; it did not exist when existing Geo2Geo
        # checkpoints were trained, so it is randomly initialised here and
        # must be learned separately (see lunar_vision.model.geo.encoder).
        self.retrieval = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,   # expects (B, S, E) – no permute needed
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
 
    # ------------------------------------------------------------------
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and flatten the spatial dimensions."""
        n, c, h, w = x.shape
        p = self.patch_size
 
        torch._assert(h == self.image_size,
                       f"Wrong image height: expected {self.image_size}, got {h}")
        torch._assert(w == self.image_size,
                       f"Wrong image width: expected {self.image_size}, got {w}")
 
        n_h, n_w = h // p, w // p
 
        # (B, C, H, W) → (B, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
 
        # (B, hidden_dim, n_h, n_w) → (B, hidden_dim, n_h*n_w)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
 
        # (B, hidden_dim, S) → (B, S, hidden_dim)   [batch_first=True]
        x = x.permute(0, 2, 1)
 
        return x
 
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            latent: (B, num_patches, hidden_dim)
        """
        x = self._process_input(x)       # (B, S, E)
        x = x + self.pos_embed           # add positional info
        latent = self.transformer(x)     # (B, S, E)
        return latent

    # ------------------------------------------------------------------
    def encode_retrieval(self, x: torch.Tensor) -> torch.Tensor:
        """CLIP-facing retrieval path: one vector per image.

        Unlike forward(), this accepts any input resolution divisible by
        patch_size (not just image_size) — e.g. CLIP patches are 512x512
        while this encoder was trained at image_size=256 — by resizing
        pos_embed to the actual patch grid via bicubic interpolation, then
        prepending the retrieval token and running the transformer.

        Args:
            x: (B, C, H, W)
        Returns:
            (B, hidden_dim) retrieval vector (token 0 of the output sequence).
        """
        n, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"Input height/width must be divisible by patch_size ({p}). "
                f"Got {(h, w)}."
            )

        x = self.conv_proj(x)                             # (B, hidden_dim, gh, gw)
        _, hidden_dim, grid_h, grid_w = x.shape
        x = x.reshape(n, hidden_dim, grid_h * grid_w).permute(0, 2, 1)  # (B, S, E)

        pos_embed = self._resized_pos_embed((grid_h, grid_w))
        x = x + pos_embed.to(device=x.device, dtype=x.dtype)

        retrieval_tokens = self.retrieval.to(device=x.device, dtype=x.dtype).expand(n, -1, -1)
        x = torch.cat([retrieval_tokens, x], dim=1)        # (B, 1 + S, E)

        tokens = self.transformer(x)
        return tokens[:, 0, :]

    # ------------------------------------------------------------------
    def _resized_pos_embed(self, target_grid_size: tuple[int, int]) -> torch.Tensor:
        """Resize pos_embed to target_grid_size if it differs from the grid
        it was trained at (e.g. 16x16 -> 32x32 when moving from this
        encoder's native 256x256 training tile to CLIP's 512x512 patch with
        the same patch_size)."""
        source_grid_size = source_pos_grid_size(
            token_count=int(self.pos_embed.shape[1]),
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        return resize_pos_embed(
            self.pos_embed,
            source_grid_size=source_grid_size,
            target_grid_size=target_grid_size,
        )


# ──────────────────────────────────────────────
# DECODER
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Convolutional up-sampling decoder.
 
    Takes the latent patch sequence and reconstructs the original image
    through a 2-D reshape followed by a hierarchical ConvTranspose2d stack.
 
    Args:
        patch_size   (int): Must match the encoder's patch_size.  Default 16.
        image_size   (int): Target output resolution.             Default 256.
        img_channels (int): Output channels.                      Default 3.
        hidden_dim   (int): Must match the encoder's hidden_dim.  Default 512.
    """
 
    def __init__(
        self,
        patch_size: int = 16,
        image_size: int = 256,
        img_channels: int = 3,
        hidden_dim: int = 512,
    ):
        super().__init__()
 
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size   # spatial side of the patch grid
        self.hidden_dim = hidden_dim
 
        # ── Bottleneck projection ──────────────────────────────────────
        # Reduce channel depth before upsampling for efficiency.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
 
        # ── Progressive up-sampling ────────────────────────────────────
        #   grid_size (16) → 64 (×4) → 256 (×4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
 
        # ── Final refinement + channel projection ─────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, img_channels, kernel_size=1),
            # Tanh → pixel values in [-1, 1] (matches normalisation in Dataset)
            nn.Tanh(),
        )
 
    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, num_patches, hidden_dim)
        Returns:
            recon: (B, C, H, W)
        """
        n, s, d = z.shape
        g = self.grid_size
 
        # (B, S, E) → (B, g, g, E)
        z = z.reshape(n, g, g, d)
 
        # (B, g, g, E) → (B, E, g, g)   Conv2d format
        z = z.permute(0, 3, 1, 2)
 
        z = self.bottleneck(z)   # (B, 256, g, g)
        z = self.up1(z)          # (B, 128, g*4, g*4)
        z = self.up2(z)          # (B, 64,  H, W)
        recon = self.head(z)     # (B, C, H, W)
 
        return recon
 
 
# ──────────────────────────────────────────────
# FULL AUTOENCODER
# ──────────────────────────────────────────────
class Geo2Geo(nn.Module):
    """
    End-to-end Transformer Autoencoder for image reconstruction.
 
    Combines `Encoder` and `Decoder` into a single nn.Module with a
    symmetric interface:  image in → reconstructed image out.
 
    Typical usage:
        model = Geo2Geo()
        recon = model(image)          # full forward pass
        latent = model.encode(image)  # encoder only
        recon  = model.decode(latent) # decoder only
    """
 
    def __init__(
        self,
        patch_size: int = 16,
        image_size: int = 256,
        img_channels: int = 3,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()
 
        self.encoder = Encoder(
            patch_size=patch_size,
            image_size=image_size,
            img_channels=img_channels,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_layers=num_layers,
        )
 
        self.decoder = Decoder(
            patch_size=patch_size,
            image_size=image_size,
            img_channels=img_channels,
            hidden_dim=hidden_dim,
        )
 
        self._init_weights()
 
    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier / normal initialisation for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
 
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) normalised to [-1, 1]
        Returns:
            recon: (B, C, H, W) in [-1, 1]
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon
 
 
# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
 
    model = Geo2Geo(
        patch_size=16,
        image_size=256,
        img_channels=3,
        hidden_dim=512,
        nheads=8,
        num_layers=6,
    ).to(device)
 
    dummy = torch.randn(4, 3, 256, 256, device=device)
    recon = model(dummy)
 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {recon.shape}")
    print(f"Params : {total_params:,}")
 
