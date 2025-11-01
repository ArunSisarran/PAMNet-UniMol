import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops, to_dense_batch


class Hybrid_Model(nn.Module):

    def __init__(self, pamnet_model, unimol_dim=512, fusion_dim=256, num_heads=2, dropout=0.1, freeze_pamnet=False):
        super().__init__()
        
        self.pamnet_model = pamnet_model
        self.freeze_pamnet = freeze_pamnet
        
        if freeze_pamnet:
            for param in self.pamnet_model.parameters():
                param.requires_grad = False
        
        pamnet_dim = pamnet_model.dim  # 128
        
        self.pamnet_proj = nn.Linear(pamnet_dim, fusion_dim)
        self.unimol_proj = nn.Linear(unimol_dim, fusion_dim)
        
        # Cross-attention modules
        self.cross_attn_p2u = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_u2p = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.norm3 = nn.LayerNorm(fusion_dim * 2)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),     
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),    
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1) l
        )
        
        self._init_predictor_weights()
        self.dropout = nn.Dropout(dropout)
    
    def _init_predictor_weights(self):
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_output_bias(self, target_mean):
        final_layer = self.predictor[-1]
        if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, target_mean)
            print(f"  Set output bias to target mean: {target_mean:.4f}")

    def extract_pamnet_features(self, data):
        x_raw = data.x
        batch = data.batch
        pos = data.pos
        edge_index_l = data.edge_index
        
        x = torch.index_select(self.pamnet_model.embeddings, 0, x_raw.long())
        
        row, col = radius(pos, pos, self.pamnet_model.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()
        
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()
        
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = \
            self.pamnet_model.indices(edge_index_l, num_nodes=x.size(0))
        
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_kj).norm(dim=-1)
        angle2 = torch.atan2(b, a)
        
        pos_i_pair, pos_j1_pair, pos_j2_pair = pos[idx_i_pair], pos[idx_j1_pair], pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.linalg.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
        angle1 = torch.atan2(b, a)
        
        rbf_l = self.pamnet_model.rbf_l(dist_l)
        rbf_g = self.pamnet_model.rbf_g(dist_g)
        sbf1 = self.pamnet_model.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.pamnet_model.sbf(dist_l, angle2, idx_kj)
        
        edge_attr_rbf_l = self.pamnet_model.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.pamnet_model.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.pamnet_model.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.pamnet_model.mlp_sbf2(sbf2)
        
        for layer in range(self.pamnet_model.n_layer):
            x, _, _ = self.pamnet_model.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            x, _, _ = self.pamnet_model.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf2, edge_attr_sbf1,
                                                            idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l)
        
        return x, batch

    def forward(self, graph_data, unimol_embeddings, return_attention=False):
        # Extract node-level features from PAMNet
        if self.freeze_pamnet:
            with torch.no_grad():
                pamnet_node_features, batch = self.extract_pamnet_features(graph_data)
        else:
            pamnet_node_features, batch = self.extract_pamnet_features(graph_data)
        
        if unimol_embeddings.dim() == 1:
            unimol_embeddings = unimol_embeddings.unsqueeze(0)
        
        pamnet_proj = self.pamnet_proj(pamnet_node_features) 
        unimol_expanded = unimol_embeddings[batch] 
        unimol_proj = self.unimol_proj(unimol_expanded) 
        
        pamnet_dense, mask = to_dense_batch(pamnet_proj, batch)  
        unimol_dense, _ = to_dense_batch(unimol_proj, batch)     
        
        pamnet_attended, attn_p2u = self.cross_attn_p2u(
            query=pamnet_dense,
            key=unimol_dense,
            value=unimol_dense,
            key_padding_mask=~mask,  
            need_weights=return_attention
        )
        pamnet_attended = self.norm1(pamnet_dense + self.dropout(pamnet_attended))
        
        unimol_attended, attn_u2p = self.cross_attn_u2p(
            query=unimol_dense,
            key=pamnet_dense,
            value=pamnet_dense,
            key_padding_mask=~mask,
            need_weights=return_attention
        )
        unimol_attended = self.norm2(unimol_dense + self.dropout(unimol_attended))
        
        pamnet_attended_nodes = pamnet_attended[mask]  
        unimol_attended_nodes = unimol_attended[mask]  
        
        fused = torch.cat([pamnet_attended_nodes, unimol_attended_nodes], dim=-1)  
        
        fused = self.norm3(fused + self.ffn(fused))  
        
        fused_pooled = global_add_pool(fused, batch) 
        
        output = self.predictor(fused_pooled).squeeze(-1)  
        
        if return_attention:
            attention_info = {
                'pamnet_to_unimol': attn_p2u,
                'unimol_to_pamnet': attn_u2p
            }
            return output, attention_info
        
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
