import torch 
from pathlib import Path
import sys
SRC_DIR = Path(__file__).resolve().parent / "src"

from lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from lunar_clip.model.lunar_clip_model import LunarCLIPModel

if __name__ == "__main__":
        
    print("loading text adapter")
    text_adapter = LunarTextEncoder(
        encoder="wordpiece",
        tokenizer_path="artifacts/tokenizers/wordpiece/v3.0/tokenizer.json",
        checkpoint_path="artifacts/text_models/wordpiece/wordpiece_mlm-epoch=39-train_loss=0.583.ckpt",
        freeze_encoder=True,
    )
    print("success loading text adapter")
    print("loading vision adapter")
    vision_adapter = LunarVisionEncoder(
        encoder="wac",
        checkpoint_path="artifacts/vision_models/wac2wac/best.ckpt",
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