import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid_Model(nn.Module):
    def __init__(self, pamnet_dim=256, unimol_dim=512, fusion_dim=256, num_heads=4):
        super().__init__()

        self.pamnet_proj = nn.Linear(pamnet_dim, fusion_dim)
        self.unimol_proj = nn.Linear(unimol_dim, fusion_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        self.ffself.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(0.1)
        )
        
        self.dropout = nn.Dropout(0.1)
