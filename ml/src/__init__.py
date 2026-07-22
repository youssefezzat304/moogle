from pathlib import Path
import sys

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from lunar_clip.contracts.batches import RetrievalBatch
from lunar_clip.contracts.outputs import LunarCLIPOutput
from lunar_clip.data.clip_dataset import LunarCLIPDataset
from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.losses.contrastive import symmetric_contrastive_loss
from lunar_clip.model.lunar_clip_model import LunarCLIPModel
from lunar_clip.retrieval.metrics import full_index_retrieval_metrics
from lunar_clip.retrieval.vector_store import InMemoryVectorStore
from lunar_clip.training.metrics import in_batch_top1_retrieval_metrics
from lunar_clip.training.train_clip import (
    LunarCLIPDataModule,
    LunarCLIPLightningModule,
    LunarCLIPTrainingConfig,
    LunarCLIPTrainingResult,
    train_clip,
)
from lunar_clip.utils import (
    VISION_ENCODERS,
    build_vision_adapter,
)
from lunar_text.model.bpe.config import ModelConfig
from lunar_text.model.bpe.model import BPELunarMLM
from lunar_text.tokenizer.bpe.wrapper import BPETokenizerWrapper
from lunar_text.training.bpe.masking import Masking
from lunar_text.utils.tokenizers import (
    REQUIRED_SPECIAL_TOKENS,
    validate_required_special_tokens,
)

__all__ = [
    "BPETokenizerWrapper",
    "InMemoryVectorStore",
    "LunarCLIPDataModule",
    "LunarCLIPDataset",
    "LunarCLIPLightningModule",
    "LunarCLIPModel",
    "LunarCLIPOutput",
    "LunarCLIPTrainingConfig",
    "LunarCLIPTrainingResult",
    "LunarTextEncoder",
    "LunarVisionEncoder",
    "LunarGeoData",
    "BPELunarMLM",
    "Masking",
    "ModelConfig",
    "REQUIRED_SPECIAL_TOKENS",
    "RetrievalBatch",
    "VISION_ENCODERS",
    "build_vision_adapter",
    "full_index_retrieval_metrics",
    "in_batch_top1_retrieval_metrics",
    "symmetric_contrastive_loss",
    "train_clip",
    "validate_required_special_tokens",
]


def __getattr__(name: str):
    if name == "LunarGeoData":
        from lunar_data.lunar_geo_data import LunarGeoData

        return LunarGeoData
    raise AttributeError(f"module 'src' has no attribute {name!r}")
