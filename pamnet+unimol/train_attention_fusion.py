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
import time
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

from attention_fusion_model import Attention_Fusion


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


def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    model.eval()
    
    with torch.no_grad():
        for data, unimol_embeddings in loader:
            data = data.to(device)
            unimol_embeddings = unimol_embeddings.to(device)
            
            output = model(data, unimol_embeddings)
            
            target = data.y.view(-1)
            output = output.view(-1)
            
            batch_mae = (output - target).abs().sum().item()
            mae += batch_mae
    
    ema.resume(model)
    return mae / len(loader.dataset)


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
    parser.add_argument('--fusion_dim', type=int, default=128, help='Fusion layer dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pamnet_checkpoint', type=str, default='best_model.pt',
                       help='Path to pretrained PAMNet')
    parser.add_argument('--freeze_pamnet', action='store_true',
                       help='Freeze PAMNet weights (default: False)')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of PAMNet layers')
    parser.add_argument('--dim', type=int, default=128, help='PAMNet hidden dimension')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='PAMNet local cutoff')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='PAMNet global cutoff')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--norm', type=int, default=1000, help="Norm amount")
    parser.add_argument('--warmup', type=int, default=1, help='warmup scheduler epochs')
    parser.add_argument('--scheduler', type=str, default='exponential', choices=['exponential', 'cosine'])
    parser.add_argument('--gamma', type=float, default=0.9961697, help='scheduler gamma')
    parser.add_argument('--tmax', type=int, default=10, help='T_max for cosine annealing lr')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='eta_min for cosine annealing')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--wandb_project', type=str, default='Attention_Fusion',
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
                "architecture": "SimpleFusion",
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
                "norm": args.norm,
                "warmup": args.warmup,
                "gamma": args.gamma
            },
            name=f"simple_fusion_target{args.target}_lr{args.lr}_heads{args.num_heads}_{args.fusion_dim}fdim_{args.dim}dim_{args.batch_size}batch"
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
    dataset = QM9(path, transform=MyTransform())

    emb_path = osp.join('.', 'data', args.dataset, 'precomputed', 'unimol_embeddings.pt')
    emb_data = torch.load(emb_path)
    all_embeddings = emb_data['embeddings']

    split_path = osp.join('.', 'data', args.dataset, 'split_indices.pt')

    split_data = torch.load(split_path)
    train_indices = split_data['train_indices']
    val_indices = split_data['val_indices']
    test_indices = split_data['test_indices']

    print(f"Using fixed split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_embeddings = all_embeddings[train_indices]
    val_embeddings = all_embeddings[val_indices]
    test_embeddings = all_embeddings[test_indices]

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

    config = Config(
        dataset=args.dataset,
        dim=args.dim,
        n_layer=args.n_layer,
        cutoff_l=args.cutoff_l,
        cutoff_g=args.cutoff_g
    )
    
    try:
        pamnet_model = PAMNet(config).to(device)
        pamnet_model.load_state_dict(torch.load(args.pamnet_checkpoint, map_location=device))
    except Exception as e:
        print(f"ERROR loading PAMNet: {e}")
        return

    model = Attention_Fusion(
        pamnet_model=pamnet_model,
        unimol_dim=unimol_dim,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_pamnet=args.freeze_pamnet
    ).to(device)
    
    trainable_params = count_parameters(model)
    
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

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)

    if args.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.eta_min, last_epoch=-1)

    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup,
        after_scheduler=scheduler
    )

    ema = EMA(model, decay=0.999)
    
    save_folder = osp.join(".", "save", args.dataset + "atten_fusion")
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    
    print("Start training!")
    
    best_val_loss = None
    test_loss_ema = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
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
            clip_grad_norm_(model.parameters(), max_norm=args.norm, norm_type=2)
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
        val_loss_ema = test(model, val_loader, ema, device)
        train_loss_ema = test(model, train_loader, ema, device)
        
        if best_val_loss is None or val_loss_ema <= best_val_loss:
            test_loss_ema = test(model, test_loader, ema, device)
            best_val_loss = val_loss_ema
            patience_counter = 0

            torch.save(model.state_dict(), osp.join(save_folder, "best_fusion_qm9.pt"))
            
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {args.patience} epochs)")
            break
        
        current_lr = optimizer.param_groups[0]['lr']

        gate_value = torch.sigmoid(model.gate_param).item()

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/mae": loss,
                "val/mae": val_loss_ema,
                "test/mae": test_loss_ema if test_loss_ema is not None else 0.0,
                "train/learning_rate": current_lr,
            })

        print('Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch+1, train_loss_ema, val_loss_ema, test_loss_ema))
        print(f"PAMNet scale: {model.pamnet_scale.item():.4f}")
        print(f"UniMol scale: {model.unimol_scale.item():.4f}")
        print(f"Gate Value (0=PAMNet, 1=Fusion): {gate_value:.6f}")
        print(f"Ratio (PAMNet/UniMol): {(model.pamnet_scale / model.unimol_scale).item():.2f}")

    print('Best Validation MAE:', best_val_loss)
    print('Testing MAE:', test_loss_ema)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
