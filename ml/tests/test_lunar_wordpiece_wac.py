import torch 
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

print("Using:", SRC_DIR)
from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.model.lunar_clip_model import LunarCLIPModel

if __name__ == "__main__":
    print("loading text adapter")
    text_adapter = LunarTextEncoder(
        encoder="wordpiece",
        tokenizer_path="artifacts/tokenizers/wordpiece/v3.0/tokenizer.json",
        checkpoint_path="artifacts/text_models/wordpiece/wordpiece_mlm-epoch=98-train_loss=0.454.ckpt",
        freeze_encoder=True,
    )
    print("success loading text adapter")
    print("loading vision adapter")
    vision_adapter = LunarVisionEncoder(
        encoder="geo",
        checkpoint_path="artifacts/vision_models/geo2geo/best.ckpt",
        freeze_encoder=True,
    )
    print("success loading vision adapter")
    print("loading lunar clip model")
    model = LunarCLIPModel(
        text_adapter=text_adapter,
        vision_adapter=vision_adapter,
        projection_dim=512,
        temperature=0.07,
    )
    print("success loading lunar clip model")