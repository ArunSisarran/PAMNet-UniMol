import os
import sys
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import csv
import time
from datetime import datetime
import h5py

current_dir = os.path.dirname(os.path.abspath(__file__))
pamnet_dir = os.path.join(os.path.dirname(current_dir), "Physics-aware-Multiplex-GNN")
sys.path.append(pamnet_dir)

from models import PAMNet, PAMNet_s, Config
sys.path.append(os.path.join(pamnet_dir, "utils"))
from ema import EMA

sys.path.append(os.path.join(pamnet_dir, "datasets"))
from qm9_dataset import QM9
from data_processing import DataProcessing


class QM9WithEmbeddings(torch.utils.data.Dataset):
    
    def __init__(self, qm9_dataset, embeddings):
        self.qm9_dataset = qm9_dataset
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.qm9_dataset)
    
    def __getitem__(self, idx):
        return self.qm9_dataset[idx], self.embeddings[idx]


def collate_with_embeddings(batch):
    """Custom collate function to handle (data, embedding) tuples"""
    from torch_geometric.data import Batch
    
    data_list, embeddings = zip(*batch)
    
    batched_data = Batch.from_data_list(data_list)
    
    batched_embeddings = torch.stack(embeddings)
    
    return batched_data, batched_embeddings


class SimpleHybridModel(nn.Module):
    """
    Hybrid model: Use PAMNet's learned node representations + UniMol molecular embeddings
    """
    def __init__(self, pamnet_model, unimol_dim=512, freeze_pamnet=False):
        super(SimpleHybridModel, self).__init__()
        self.pamnet_model = pamnet_model
        self.freeze_pamnet = freeze_pamnet
        
        if freeze_pamnet:
            for param in self.pamnet_model.parameters():
                param.requires_grad = False
        
        pamnet_dim = pamnet_model.dim  # 128
        
        concat_dim = pamnet_dim + unimol_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def extract_pamnet_features(self, data):
        """
        Run PAMNet forward pass and extract final node embeddings before pooling
        """
        from torch_geometric.nn import global_add_pool, radius
        from torch_geometric.utils import remove_self_loops
        
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
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle2 = torch.atan2(b, a)
        
        pos_i_pair, pos_j1_pair, pos_j2_pair = pos[idx_i_pair], pos[idx_j1_pair], pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
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
        
        pamnet_features = global_add_pool(x, batch)
        
        return pamnet_features
        
    def forward(self, graph_data, unimol_embeddings):
        if self.freeze_pamnet:
            with torch.no_grad():
                pamnet_features = self.extract_pamnet_features(graph_data)
        else:
            pamnet_features = self.extract_pamnet_features(graph_data)

        if unimol_embeddings.dim() == 1:
            unimol_embeddings = unimol_embeddings.unsqueeze(0)

        combined = torch.cat([pamnet_features, unimol_embeddings], dim=-1)

        output = self.fusion(combined)

        output = output.view(-1)

        return output  


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_log(log_path, epoch, train_mae, val_mae, test_mae, epoch_time, lr, is_best=False):
    """Save training metrics to CSV file"""
    file_exists = osp.exists(log_path)
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'epoch', 'train_mae', 'val_mae', 'test_mae', 
                'epoch_time_sec', 'learning_rate', 'is_best'
            ])
        
        # Write metrics
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch,
            f'{train_mae:.7f}',
            f'{val_mae:.7f}',
            f'{test_mae:.7f}',
            f'{epoch_time:.2f}',
            f'{lr:.8f}',
            is_best
        ])


def test(model, loader, ema, device, use_precomputed=False, data_processor=None):
    """Evaluate model on a dataset with progress bar"""
    mae = 0
    ema.assign(model)
    model.eval()
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch_item in pbar:
            if use_precomputed:
                data, unimol_embeddings = batch_item
                data = data.to(device)
                unimol_embeddings = unimol_embeddings.to(device)
            else:
                data = batch_item
                data = data.to(device)
                
                smiles_list = data.smiles if hasattr(data, 'smiles') else []
                
                if len(smiles_list) == 0:
                    continue
                
                unimol_result = data_processor.data_process_unimol(smiles_list)
                unimol_embeddings = data_processor.extract_unimol_embeddings(unimol_result)
                
                if unimol_embeddings is None:
                    continue
                    
                unimol_embeddings = unimol_embeddings.to(device)
            
            output = model(data, unimol_embeddings)
            
            # Ensure output and target have same shape
            target = data.y.view(-1)
            output = output.view(-1)
            
            batch_mae = (output - target).abs().sum().item()
            mae += batch_mae
            
            pbar.set_postfix({'MAE': f'{batch_mae/len(target):.4f}'})
    
    ema.resume(model)
    return mae / len(loader.dataset)


def load_pamnet_checkpoint(checkpoint_path, config):
    """
    Load PAMNet model from checkpoint file (PyTorch .pt/.h5 format or actual HDF5)
    """
    print(f"Loading PAMNet from checkpoint: {checkpoint_path}")
    
    model = PAMNet(config)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Loaded as PyTorch checkpoint")
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Found 'model_state_dict' with {len(state_dict)} parameters")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"Found 'state_dict' with {len(state_dict)} parameters")
            else:
                state_dict = checkpoint
                print(f"Using entire checkpoint as state_dict ({len(state_dict)} parameters)")
        else:
            print(f"ERROR: Checkpoint is not a dictionary")
            raise ValueError("Invalid checkpoint format")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys missing from checkpoint")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
        
        return model
        
    except Exception as e:
        print(f"Failed to load as PyTorch checkpoint: {e}")
        print("Trying as HDF5 format...")
        
        try:
            with h5py.File(checkpoint_path, 'r') as f:
                print(f"HDF5 file contains {len(f.keys())} top-level keys")
                
                state_dict = {}
                
                def load_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight = torch.from_numpy(obj[()])
                        state_dict[name] = weight
                
                f.visititems(load_weights)
            
            print(f"Loaded {len(state_dict)} parameters from HDF5")
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys missing from checkpoint")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
            
            return model
            
        except Exception as e2:
            print(f"Failed to load as HDF5: {e2}")
            raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=480, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset to be used')
    parser.add_argument('--model', type=str, default='PAMNet', choices=['PAMNet', 'PAMNet_s'])
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--target', type=int, default=7, help='Index of target for prediction')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='cutoff in global layer')
    parser.add_argument('--unimol_model', type=str, default='unimolv2', help='UniMol model name')
    parser.add_argument('--unimol_size', type=str, default='84m', help='UniMol model size')
    parser.add_argument('--use_precomputed', action='store_true', help='Use pre-computed UniMol embeddings')
    parser.add_argument('--pamnet_checkpoint', type=str, default=None, 
                       help='Path to pretrained PAMNet checkpoint file (.pt, .h5, .pth)')
    parser.add_argument('--freeze_pamnet', action='store_true',
                       help='Freeze PAMNet weights during training (only train fusion layers)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    # ==================== LOAD DATASET ====================
    print("Loading QM9 dataset...")
    path = osp.join('.', 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform())

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    # ==================== HANDLE EMBEDDINGS ====================
    if args.use_precomputed:
        emb_path = osp.join('.', 'data', args.dataset, 'precomputed', 'unimol_embeddings.pt')
        
        if not osp.exists(emb_path):
            print(f"\nERROR: Pre-computed embeddings not found at {emb_path}")
            print("Please run: python precompute_unimol_embeddings.py")
            return
        
        print(f"Loading pre-computed embeddings from {emb_path}...")
        emb_data = torch.load(emb_path)
        all_embeddings = emb_data['embeddings']
        print(f"Loaded embeddings: {all_embeddings.shape}")
        
        train_embeddings = all_embeddings[:110000]
        val_embeddings = all_embeddings[110000:120000]
        test_embeddings = all_embeddings[120000:]
        
        train_dataset = QM9WithEmbeddings(train_dataset, train_embeddings)
        val_dataset = QM9WithEmbeddings(val_dataset, val_embeddings)
        test_dataset = QM9WithEmbeddings(test_dataset, test_embeddings)
        
        unimol_dim = all_embeddings.shape[-1]
        print(f"UniMol embedding dimension: {unimol_dim}")
        
        data_processor = None  
        
    else:
        print("\nInitializing UniMol for on-the-fly embedding computation...")
        print("Note: Training will be MUCH slower. Consider using --use_precomputed")
        
        data_processor = DataProcessing(
            pamnet_model=None,
            unimol_model=args.unimol_model,
            unimol_model_size=args.unimol_size
        )
        
        print("Determining UniMol embedding dimension...")
        sample_result = data_processor.data_process_unimol(["CCO"])
        sample_emb = data_processor.extract_unimol_embeddings(sample_result)
        unimol_dim = sample_emb.shape[-1]
        print(f"UniMol embedding dimension: {unimol_dim}")

    if args.use_precomputed:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 collate_fn=collate_with_embeddings)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_with_embeddings)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_with_embeddings)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Data loaded! Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ==================== INITIALIZE MODELS ====================
    print("\nInitializing PAMNet...")
    config = Config(
        dataset=args.dataset, 
        dim=args.dim, 
        n_layer=args.n_layer, 
        cutoff_l=args.cutoff_l, 
        cutoff_g=args.cutoff_g
    )
    
    if args.pamnet_checkpoint is not None:
        print(f"Loading pretrained PAMNet from {args.pamnet_checkpoint}...")
        try:
            pamnet_model = load_pamnet_checkpoint(args.pamnet_checkpoint, config)
            print(f"Successfully loaded pretrained PAMNet!")
        except Exception as e:
            print(f"ERROR loading pretrained PAMNet: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to fresh PAMNet initialization...")
            if args.model == 'PAMNet':
                pamnet_model = PAMNet(config)
            else:
                pamnet_model = PAMNet_s(config)
    else:
        if args.model == 'PAMNet':
            pamnet_model = PAMNet(config)
        else:
            pamnet_model = PAMNet_s(config)
    
    print(f"PAMNet initialized with {count_parameters(pamnet_model)} parameters")
    
    print("\nCreating hybrid model...")
    model = SimpleHybridModel(pamnet_model, unimol_dim=unimol_dim, freeze_pamnet=args.freeze_pamnet).to(device)
    
    total_params = count_parameters(model)
    pamnet_params = count_parameters(pamnet_model)
    fusion_params = total_params - pamnet_params
    
    if args.freeze_pamnet:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"  - PAMNet (frozen): {pamnet_params:,}")
        print(f"  - Fusion layers (trainable): {trainable_params:,}")
    else:
        print(f"Total parameters: {total_params:,}")
        print(f"  - PAMNet: {pamnet_params:,}")
        print(f"  - Fusion layers: {fusion_params:,}")
    
    # ==================== SETUP TRAINING ====================
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer, 
        multiplier=1.0, 
        total_epoch=1, 
        after_scheduler=scheduler
    )

    ema = EMA(model, decay=0.999)
    
    save_folder = osp.join(".", "save", args.dataset + "_simple")
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    # ==================== TRAINING LOOP ====================
    print("\n" + "="*50)
    print("Starting training!")
    if args.use_precomputed:
        print("Using PRE-COMPUTED embeddings (FAST)")
    else:
        print("Computing embeddings ON-THE-FLY (SLOW)")
    print("="*50)
    
    log_path = osp.join(save_folder, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"Logging training metrics to: {log_path}\n")
    
    print("DEBUG: Checking first batch...")
    for batch_item in train_loader:
        if args.use_precomputed:
            data, unimol_embeddings = batch_item
        else:
            data = batch_item
        data = data.to(device)
        if args.use_precomputed:
            unimol_embeddings = unimol_embeddings.to(device)
        
        print(f"  Batch size: {data.num_graphs}")
        print(f"  Target y shape: {data.y.shape}")
        print(f"  Target y range: [{data.y.min():.4f}, {data.y.max():.4f}]")
        print(f"  Target y mean: {data.y.mean():.4f}")
        
        with torch.no_grad():
            pamnet_out = model.pamnet_model(data)
            print(f"  PAMNet output shape: {pamnet_out.shape}")
            print(f"  PAMNet output range: [{pamnet_out.min():.4f}, {pamnet_out.max():.4f}]")
            
            output = model(data, unimol_embeddings)
            print(f"  Final output shape: {output.shape}")
            print(f"  Final output range: [{output.min():.4f}, {output.max():.4f}]")
        break
    print("")
    
    best_val_loss = None
    test_loss = None
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        loss_all = 0
        step = 0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_item in pbar:
            if args.use_precomputed:
                data, unimol_embeddings = batch_item
                data = data.to(device)
                unimol_embeddings = unimol_embeddings.to(device)
            else:
                data = batch_item
                data = data.to(device)
                
                smiles_list = data.smiles if hasattr(data, 'smiles') else []
                
                if len(smiles_list) == 0:
                    continue
                
                unimol_result = data_processor.data_process_unimol(smiles_list)
                unimol_embeddings = data_processor.extract_unimol_embeddings(unimol_result)
                
                if unimol_embeddings is None:
                    continue
                    
                unimol_embeddings = unimol_embeddings.to(device)
            
            optimizer.zero_grad()
            
            output = model(data, unimol_embeddings)
            
            # Ensure shapes match
            target = data.y.view(-1)
            output = output.view(-1)
            
            loss = F.l1_loss(output, target)
            loss_all += loss.item() * data.num_graphs
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()

            curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
            scheduler_warmup.step(curr_epoch)

            ema(model)
            step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
        train_loss = loss_all / len(train_loader.dataset)
        val_loss = test(model, val_loader, ema, device, args.use_precomputed, data_processor)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        is_best = False
        if best_val_loss is None or val_loss <= best_val_loss:
            test_loss = test(model, test_loader, ema, device, args.use_precomputed, data_processor)
            best_val_loss = val_loss
            is_best = True
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'test_loss': test_loss,
                'config': vars(args),
            }, osp.join(save_folder, "best_simple_model.pt"))
        
        save_training_log(
            log_path, 
            epoch + 1, 
            train_loss, 
            val_loss, 
            test_loss if test_loss is not None else 0.0,
            epoch_time,
            current_lr,
            is_best
        )

        print(f'Epoch {epoch+1:03d} [{epoch_time:.1f}s]: '
              f'Train MAE: {train_loss:.7f}, Val MAE: {val_loss:.7f}, '
              f'Test MAE: {test_loss:.7f}' + (' BEST!' if is_best else ''))
    
    # ==================== FINAL RESULTS ====================
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f'Best Validation MAE: {best_val_loss:.7f}')
    print(f'Final Test MAE: {test_loss:.7f}')
    print(f'\nTraining log saved to: {log_path}')
    print(f'Best model saved to: {osp.join(save_folder, "best_simple_model.pt")}')


if __name__ == "__main__":
    main()