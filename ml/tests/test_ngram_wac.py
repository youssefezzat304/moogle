import torch

from src.lunar_clip.encoders.text.lunar_text_encoder import LunarTextEncoder
from src.lunar_clip.encoders.vision.lunar_vision_encoder import LunarVisionEncoder
from src.lunar_clip.model.lunar_clip_model import LunarCLIPModel 



if __name__ == "__main__":
        
    text_adapter = LunarTextEncoder(
        encoder="ngram",
        tokenizer_path="artifacts/tokenizers/ngram/v1.0/tokenizer.json",
        checkpoint_path="artifacts/text_models/ngram/checkpoint_best.pth",
        freeze_encoder=True,
    )

    vision_adapter = LunarVisionEncoder(
        encoder="wac",
        checkpoint_path="artifacts/vision_models/wac2wac/best.ckpt",
        freeze_encoder=True,
    )

    model = LunarCLIPModel(
        text_adapter=text_adapter,
        vision_adapter=vision_adapter,
        projection_dim=512,
        temperature=0.07,
    )
    
    model.eval()

    text = "a large lunar crater on the surface"

    with torch.no_grad():
        text_output = text_adapter.encode_text(text)

    print("Text vector shape:")
    print(text_output.vectors.shape)

    print("\nFirst 10 values:")
    print(text_output.vectors[0][:10])