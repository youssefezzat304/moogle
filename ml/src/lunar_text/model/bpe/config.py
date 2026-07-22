import json
from dataclasses import asdict, dataclass

from lunar_apps.dashboard.config import CONFIG


MLM_CONFIG = CONFIG.get("mlm") or {}
MLM_MODEL_CONFIG = MLM_CONFIG.get("model") or {}
MLM_TRAINING_CONFIG = MLM_CONFIG.get("training") or {}


def _model_default(key: str, fallback):
    return MLM_MODEL_CONFIG.get(key, fallback)


def _training_default(key: str, fallback):
    return MLM_TRAINING_CONFIG.get(key, fallback)


@dataclass
class ModelConfig:
    vocab_size: int = 8596
    embed_dim: int = _model_default("embed_dim", 512)
    num_heads: int = _model_default("num_heads", 8)
    num_layers: int = _model_default("num_layers", 6)
    ffn_dim: int = _model_default("ffn_dim", 1024)
    dropout: float = _model_default("dropout", 0.1)
    max_seq_len: int = _training_default("max_seq_len", 512)
    pad_token_id: int = 0
    layer_norm_eps: float = _model_default("layer_norm_eps", 1e-5)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))
