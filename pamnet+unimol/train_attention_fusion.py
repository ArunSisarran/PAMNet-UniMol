import os
import sys
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
import csv
from datetime import datetime
import h5py
from torch_geometric.data import Batch
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
pamnet_dir = os.path.join(os.path.dirname(current_dir), "Physics-aware-Multiplex-GNN")
sys.path.append(pamnet_dir)

from models import PAMNet, Config
sys.path.append(os.path.join(pamnet_dir, "utils"))
from ema import EMA

sys.path.append(os.path.join(pamnet_dir, "datasets"))
from qm9_dataset import QM9

from attention_fusion_model import Hybrid_Model


class QM9WithEmbeddings(torch.utils.data.Dataset):
    
    def __init__(self, qm9_dataset, embeddings):
        self.qm9_dataset = qm9_dataset
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.qm9_dataset)
    
    def __getitem__(self, idx):
        return self.qm9_dataset[idx], self.embeddings[idx]


def collate_with_embeddings(batch):
    
    data_list, embeddings = zip(*batch)
    batched_data = Batch.from_data_list(data_list)
    batched_embeddings = torch.stack(embeddings)
    
    return batched_data, batched_embeddings


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_training_log(log_path, epoch, train_mae, val_mae, test_mae, lr, is_best=False):
    file_exists = osp.exists(log_path)
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'timestamp', 'epoch', 'train_mae', 'val_mae', 'test_mae', 
                'learning_rate', 'is_best'
            ])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch,
            f'{train_mae:.7f}',
            f'{val_mae:.7f}',
            f'{test_mae:.7f}',
            f'{lr:.8f}',
            is_best
        ])


def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data, unimol_embeddings in loader:
        data = data.to(device)
        unimol_embeddings = unimol_embeddings.to(device)
        
        output = model(data, unimol_embeddings)
        
        target = data.y.view(-1)
        output = output.view(-1)
        
        mae += (output - target).abs().sum().item()
    
    ema.resume(model)
    return mae / len(loader.dataset)


def load_pamnet_checkpoint(checkpoint_path, config):
    print(f"Loading PAMNet from checkpoint: {checkpoint_path}")
    
    model = PAMNet(config)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Invalid checkpoint format")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        return model
        
    except Exception as e:
        try:
            with h5py.File(checkpoint_path, 'r') as f:
                print(f"HDF5 file contains {len(f.keys())} top-level keys")
                
                state_dict = {}
                
                def load_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight = torch.from_numpy(obj[()])
                        state_dict[name] = weight
                
                f.visititems(load_weights)
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            return model
            
        except Exception as e2:
            raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train attention fusion model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=480, help='Random seed')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--target', type=int, default=7, help='Target property index')
    parser.add_argument('--fusion_dim', type=int, default=256, help='Fusion layer dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pamnet_checkpoint', type=str, default='best_model.h5',
                       help='Path to pretrained PAMNet checkpoint (.h5)')
    parser.add_argument('--freeze_pamnet', action='store_true',
                       help='Freeze PAMNet weights (default: False)')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of PAMNet layers')
    parser.add_argument('--dim', type=int, default=256, help='PAMNet hidden dimension')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='PAMNet local cutoff')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='PAMNet global cutoff')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--wandb_project', type=str, default='pamnet-unimol-fusion',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='arunsisarrancs-hunter-college',
                       help='WandB entity (username or team)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "architecture": "AttentionFusion",
                "dataset": args.dataset,
                "target": args.target,
                "learning_rate": args.lr,
                "weight_decay": args.wd,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "n_layer": args.n_layer,
                "dim": args.dim,
                "cutoff_l": args.cutoff_l,
                "cutoff_g": args.cutoff_g,
                "fusion_dim": args.fusion_dim,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "freeze_pamnet": args.freeze_pamnet,
                "patience": args.patience,
                "seed": args.seed,
            },
            name=f"attention_target{args.target}_lr{args.lr}_heads{args.num_heads}_{args.fusion_dim}fdim_{args.dim}dim"
        )
    
    print(f"Attention Fusion Training - Target: {args.target}, LR: {args.lr}, Freeze PAMNet: {args.freeze_pamnet}")
    
    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    path = osp.join('.', 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform()).shuffle()

    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    emb_path = osp.join('.', 'data', args.dataset, 'precomputed', 'unimol_embeddings.pt')
    
    if not osp.exists(emb_path):
        print(f"ERROR: Pre-computed embeddings not found at {emb_path}")
        return
   
    emb_data = torch.load(emb_path)
    all_embeddings = emb_data['embeddings']
    
    train_embeddings = all_embeddings[:110000]
    val_embeddings = all_embeddings[110000:120000]
    test_embeddings = all_embeddings[120000:]
    
    unimol_dim = all_embeddings.shape[-1]
    
    train_dataset = QM9WithEmbeddings(train_dataset, train_embeddings)
    val_dataset = QM9WithEmbeddings(val_dataset, val_embeddings)
    test_dataset = QM9WithEmbeddings(test_dataset, test_embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=collate_with_embeddings)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_with_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_with_embeddings)
    print("Data loaded!")

    if not osp.exists(args.pamnet_checkpoint):
        print(f"ERROR: PAMNet checkpoint not found at {args.pamnet_checkpoint}")
        return
    
    config = Config(
        dataset=args.dataset,
        dim=args.dim,
        n_layer=args.n_layer,
        cutoff_l=args.cutoff_l,
        cutoff_g=args.cutoff_g
    )
    
    try:
        pamnet_model = load_pamnet_checkpoint(args.pamnet_checkpoint, config)
    except Exception as e:
        print(f"ERROR loading PAMNet: {e}")
        return

    model = Hybrid_Model(
        pamnet_model=pamnet_model,
        unimol_dim=unimol_dim,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_pamnet=args.freeze_pamnet
    ).to(device)
    
    trainable_params = count_parameters(model)
    
    # Initialize output bias to target mean
    sample_targets = []
    for i in range(min(10000, len(train_dataset))):
        sample_targets.append(train_dataset[i][0].y.item())
    target_mean = torch.tensor(sample_targets).mean().item()
    model.set_output_bias(target_mean)
    
    print(f"Number of model parameters: {trainable_params:,}")

    if not args.no_wandb:
        wandb.config.update({
            "trainable_parameters": trainable_params,
            "target_mean": target_mean,
            "unimol_dim": unimol_dim
        })
        wandb.watch(model, log="all", log_freq=100)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=1,
        after_scheduler=scheduler
    )

    ema = EMA(model, decay=0.999)
    
    save_folder = osp.join(".", "save", args.dataset + "_attention_fusion")
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    
    log_path = osp.join(save_folder, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    print("Start training!")
    
    best_val_loss = None
    test_loss = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        loss_all = 0
        step = 0
        model.train()
        
        for data, unimol_embeddings in train_loader:
            data = data.to(device)
            unimol_embeddings = unimol_embeddings.to(device)
            
            optimizer.zero_grad()
            
            output = model(data, unimol_embeddings)
            
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

            if not args.no_wandb and step % 10 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/step": epoch * len(train_loader) + step
                })
        
        loss = loss_all / len(train_loader.dataset)
        val_loss = test(model, val_loader, ema, device)
        
        if best_val_loss is None or val_loss <= best_val_loss:
            test_loss = test(model, test_loader, ema, device)
            best_val_loss = val_loss
            patience_counter = 0
            
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'val_loss': val_loss,
            #    'test_loss': test_loss,
            #    'config': vars(args),
            #}, osp.join(save_folder, "best_attention_fusion.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break
        
        current_lr = optimizer.param_groups[0]['lr']
        is_best = (val_loss == best_val_loss)
        
        save_training_log(
            log_path,
            epoch + 1,
            loss,
            val_loss,
            test_loss if test_loss is not None else 0.0,
            current_lr,
            is_best
        )

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/mae": loss,
                "val/mae": val_loss,
                "test/mae": test_loss if test_loss is not None else 0.0,
                "train/learning_rate": current_lr,
            })

        print('Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch+1, loss, val_loss, test_loss))
    
    print('Best Validation MAE:', best_val_loss)
    print('Testing MAE:', test_loss)

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()