import torch.nn as nn

class LunarTextEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim=512):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, features):
        return self.projection(features)