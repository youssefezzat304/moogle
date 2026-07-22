import torch
import io
import base64
import numpy as np
from PIL import Image

def tensor_to_image(tensor):
    """
    Converts a PyTorch [C, H, W] tensor into a standard NumPy image.
    Handles both 3-channel (RGB) and 1-channel (Grayscale) images.
    """
    # Rearrange channels: (C, H, W) -> (H, W, C)
    img_array = tensor.permute(1, 2, 0).numpy()
    
    # Scale to 0-255 if it's normalized
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
        
    return img_array
  
def calculate_composition(geomap_tensor, legend):
    """
    Calculates the percentage coverage of each class in the patch.
    geomap_tensor: (num_classes, H, W) one-hot tensor
    legend: { "Abbrev": {"color": "#...", "description": "...", "index": 0}, ... }
    """
    composition = geomap_tensor.mean(dim=(1, 2))
    
    results = []
    for idx, percentage in enumerate(composition):
        if percentage > 0:
            match = next(((k, v) for k, v in legend.items() if v.get('index') == idx), None)
            
            if match:
                abbrev, class_info = match
                
                results.append({
                    "label": class_info['long_description'],
                    "abbrev": abbrev,
                    "color": class_info['color'],
                    "percentage": float(percentage) * 100
                })
    
    # Sort by highest percentage first
    return sorted(results, key=lambda x: x['percentage'], reverse=True)

def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert an image tensor to a base64-encoded PNG string for Ollama.

    Accepts greyscale (1, H, W) float tensors and RGB (3, H, W) uint8 tensors.
    Applies min-max normalisation defensively so out-of-range floats are safe.
    """
    tensor = tensor.float().cpu()
    t_min, t_max = tensor.min(), tensor.max()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)

    if tensor.shape[0] == 1:
        arr = (tensor.squeeze(0) * 255).byte().numpy()
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif tensor.shape[0] == 3:
        arr = (tensor.permute(1, 2, 0) * 255).byte().numpy()
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def format_composition_for_llm(composition_data):
    """
    Converts the composition dictionary into a clean, token-efficient 
    text list for the LLM prompt.
    """
    if not composition_data:
        return "No identifiable geological features in this patch."
        
    lines = []
    for data in composition_data:
        # Formats to: "- Mare Basalt (85.2% coverage, marked in #ff0000)"
        line = f"- {data['label']} ({data['percentage']:.1f}% coverage, marked in {data['color']})"
        lines.append(line)
        
    return "\n".join(lines)
