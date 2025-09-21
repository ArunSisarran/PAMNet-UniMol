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

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(0.1)
        )
        
        self.dropout = nn.Dropout(0.1)


    def forward(self, pamnet_output, unimol_output, return_attention=False):
        batch_size = pamnet_output.size(0)

        pamnet_proj = self.pamnet_proj(pamnet_output)
        unimol_proj = self.unimol_proj(unimol_output)

        pamnet_seq = pamnet_proj.unsqueeze(1)
        unimol_seq = unimol_proj.unsqueeze(1)

        combined_seq = torch.cat([pamnet_seq, unimol_seq], dim=1)

        attn_output, attention_weights = self.cross_attention(
            query=combined_seq,    
            key=combined_seq,       
            value=combined_seq,     
            need_weights=True       
        )
        
        print(f"Attention output shape: {attn_output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")

        if return_attention:
            pamnet_to_unimol = attention_weights[:, 0, 1].mean().item()
            unimol_to_pamnet = attention_weights[:, 1, 0].mean().item()
            pamnet_self = attention_weights[:, 0, 0].mean().item()
            unimol_self = attention_weights[:, 1, 1].mean().item()
            
            print(f"  PAMNet → UniMol: {pamnet_to_unimol:.3f}")
            print(f"  UniMol → PAMNet: {unimol_to_pamnet:.3f}")
            print(f"  PAMNet → PAMNet: {pamnet_self:.3f}")
            print(f"  UniMol → UniMol: {unimol_self:.3f}")

        combined_seq = self.norm1(combined_seq + self.dropout(attn_output))
        ffn_output = self.ffn(combined_seq)
        combined_seq = self.norm2(combined_seq + ffn_output)
        fused_output = combined_seq.mean(dim=1)

        print(f"Final fused output shape: {fused_output.shape}")
        
        if return_attention:
            return fused_output, attention_weights
        else:
            return fused_output

if __name__ == "__main__":
    
    fusion_model = Hybrid_Model(
        pamnet_dim=256,
        unimol_dim=512, 
        fusion_dim=256,
        num_heads=4
    )
    
    batch_size = 2
    pamnet_output = torch.randn(batch_size, 256)
    unimol_output = torch.randn(batch_size, 512)
    
    print("INPUT SHAPES:")
    print(f"  PAMNet: {pamnet_output.shape}")
    print(f"  UniMol: {unimol_output.shape}")
    print()
    
    fused_output, attention_weights = fusion_model(
        pamnet_output, 
        unimol_output, 
        return_attention=True
    )

    for batch_idx in range(batch_size):
        print(f"Molecule {batch_idx + 1}:")
        print(f"  PAMNet→PAMNet: {attention_weights[batch_idx, 0, 0]:.3f}")
        print(f"  PAMNet→UniMol: {attention_weights[batch_idx, 0, 1]:.3f}")
        print(f"  UniMol→PAMNet: {attention_weights[batch_idx, 1, 0]:.3f}")
        print(f"  UniMol→UniMol: {attention_weights[batch_idx, 1, 1]:.3f}")
        print()