"""config.py — architecture and preprocessing constants for the WAC encoder.
These must match what train_wac2wac.py used, or the checkpoint won't load
correctly / inputs won't be preprocessed the way the model expects.
"""
HIDDEN_DIM = 512
VIT_PATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_LAYERS = 6
NUM_HEADS = 8
# Normalization applied to raw [0,1] WAC pixels before they reach the encoder.
# Whoever builds the input batch (the dataset/dataloader code) must apply this.
WAC_NORMALIZE_MEAN = 0.2523528337
WAC_NORMALIZE_STD = 0.1592932492