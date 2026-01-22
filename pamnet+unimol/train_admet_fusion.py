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
from torch_geometric.loader import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import roc_auc_score, mean_absolute_error
import time
from torch_geometric.data import Batch
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
pamnet_dir = os.path.join(os.path.dirname(current_dir), "Physics-aware-Multiplex-GNN")
sys.path.append(pamnet_dir)

from models import PAMNet, Config
sys.path.append(os.path.join(pamnet_dir, "utils"))
from ema import EMA

sys.path.append(os.path.join(pamnet_dir, "datasets"))
from admet_dataset import ADMET3DDataset

from attention_fusion_model import Attention_Fusion


class ADMETWithEmbeddings(torch.utils.data.Dataset):
    def __init__(self, admet_dataset, embeddings):
        self.admet_dataset = admet_dataset
        self.embeddings = embeddings

        if len(admet_dataset) != len(embeddings):
            raise ValueError(f"Dataset size ({len(admet_dataset)}) doesn't match "
                           f"embeddings size ({len(embeddings)})")

    def __len__(self):
        return len(self.admet_dataset)

    def __getitem__(self, idx):
        return self.admet_dataset[idx], self.embeddings[idx]


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


def evaluate(model, loader, ema, device, is_classification):
    model.eval()
    ema.assign(model)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, unimol_embeddings in loader:
            data = data.to(device)
            unimol_embeddings = unimol_embeddings.to(device)

            output = model(data, unimol_embeddings)

            pred = output.view(-1)
            target = data.y.view(-1)

            if is_classification:
                pred = torch.sigmoid(pred)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    ema.resume(model)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    if is_classification:
        try:
            score = roc_auc_score(all_targets, all_preds)
        except ValueError:
            score = 0.5
        return score, "AUC"
    else:
        score = mean_absolute_error(all_targets, all_preds)
        return score, "MAE"


def main():
    parser = argparse.ArgumentParser(description='Train attention fusion model on ADMET datasets')

    # Basic settings
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='Caco2_Wang',
                       help='ADMET dataset name (e.g., Caco2_Wang, HIA_Hou, hERG)')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Task type: regression or classification')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping norm')

    # Scheduler settings
    parser.add_argument('--warmup', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['exponential', 'cosine'])
    parser.add_argument('--gamma', type=float, default=0.995, help='LR decay gamma')
    parser.add_argument('--tmax', type=int, default=50, help='T_max for cosine annealing')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Min LR for cosine')

    # PAMNet settings
    parser.add_argument('--pamnet_checkpoint', type=str, default=None,
                       help='Path to pretrained PAMNet checkpoint (optional)')
    parser.add_argument('--freeze_pamnet', action='store_true',
                       help='Freeze PAMNet weights')
    parser.add_argument('--n_layer', type=int, default=5, help='Number of PAMNet layers')
    parser.add_argument('--dim', type=int, default=128, help='PAMNet hidden dimension')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='PAMNet local cutoff')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='PAMNet global cutoff')

    # Fusion model settings
    parser.add_argument('--fusion_dim', type=int, default=128, help='Fusion layer dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # Paths
    parser.add_argument('--data_root', type=str, default='./data/admet',
                       help='Root directory for ADMET data')
    parser.add_argument('--save_dir', type=str, default='./save/admet_fusion',
                       help='Directory to save models')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='ADMET_Fusion',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='WandB entity')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging')

    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    is_classification = args.task == 'classification'

    print("=" * 70)
    print(f"ADMET Fusion Training")
    print(f"Dataset: {args.dataset} | Task: {args.task}")
    print(f"Device: {device}")
    print("=" * 70)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"fusion_{args.dataset}_{args.task}_lr{args.lr}_heads{args.num_heads}"
        )

    # Load ADMET datasets
    print("\nLoading ADMET datasets...")
    train_dataset = ADMET3DDataset(args.data_root, args.dataset, mode='train')
    val_dataset = ADMET3DDataset(args.data_root, args.dataset, mode='val')
    test_dataset = ADMET3DDataset(args.data_root, args.dataset, mode='test')

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Load precomputed UniMol embeddings
    print("\nLoading precomputed UniMol embeddings...")
    emb_dir = osp.join(args.data_root, 'processed', args.dataset, 'precomputed')

    train_emb_path = osp.join(emb_dir, 'unimol_embeddings_train.pt')
    val_emb_path = osp.join(emb_dir, 'unimol_embeddings_val.pt')
    test_emb_path = osp.join(emb_dir, 'unimol_embeddings_test.pt')

    # Check embeddings exist
    for path, split in [(train_emb_path, 'train'), (val_emb_path, 'val'), (test_emb_path, 'test')]:
        if not osp.exists(path):
            raise FileNotFoundError(
                f"Missing {split} embeddings at {path}\n"
                f"Run: python precompute_embeddings.py --dataset_type ADMET --dataset_name {args.dataset}"
            )

    train_emb_data = torch.load(train_emb_path)
    val_emb_data = torch.load(val_emb_path)
    test_emb_data = torch.load(test_emb_path)

    train_embeddings = train_emb_data['embeddings']
    val_embeddings = val_emb_data['embeddings']
    test_embeddings = test_emb_data['embeddings']

    unimol_dim = train_embeddings.shape[-1]
    print(f"  UniMol embedding dim: {unimol_dim}")

    train_dataset_wrapped = ADMETWithEmbeddings(train_dataset, train_embeddings)
    val_dataset_wrapped = ADMETWithEmbeddings(val_dataset, val_embeddings)
    test_dataset_wrapped = ADMETWithEmbeddings(test_dataset, test_embeddings)

    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_with_embeddings,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_with_embeddings,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_with_embeddings,
        num_workers=4
    )

    print("Data loaded!")

    config = Config(
        dataset=args.dataset,
        dim=args.dim,
        n_layer=args.n_layer,
        cutoff_l=args.cutoff_l,
        cutoff_g=args.cutoff_g
    )

    pamnet_model = PAMNet(config).to(device)

    if args.pamnet_checkpoint and osp.exists(args.pamnet_checkpoint):
        print(f"\nLoading pretrained PAMNet from {args.pamnet_checkpoint}")
        checkpoint = torch.load(args.pamnet_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pamnet_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            pamnet_model.load_state_dict(checkpoint, strict=False)
        print("  PAMNet weights loaded (with strict=False for compatibility)")
    else:
        print("\nNo PAMNet checkpoint provided - training from scratch")

    model = Attention_Fusion(
        pamnet_model=pamnet_model,
        unimol_dim=unimol_dim,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_pamnet=args.freeze_pamnet
    ).to(device)

    trainable_params = count_parameters(model)
    print(f"\nModel parameters: {trainable_params:,}")

    # Set output bias based on training data mean (for regression)
    if not is_classification:
        sample_targets = []
        for i in range(min(1000, len(train_dataset))):
            sample_targets.append(train_dataset[i].y.item())
        target_mean = np.mean(sample_targets)
        model.set_output_bias(target_mean)
        print(f"Target mean: {target_mean:.4f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.tmax, eta_min=args.eta_min
        )

    scheduler_warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup,
        after_scheduler=scheduler
    )

    ema = EMA(model, decay=0.999)

    if is_classification:
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = 0.0
        compare = lambda new, best: new > best
        metric_name = "AUC"
    else:
        criterion = nn.L1Loss()
        best_val_score = float('inf')
        compare = lambda new, best: new < best
        metric_name = "MAE"

    save_dir = osp.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.config.update({
            "trainable_parameters": trainable_params,
            "unimol_dim": unimol_dim,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset)
        })

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    best_epoch = 0
    patience_counter = 0
    test_score_at_best = None

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        loss_all = 0
        step = 0

        for data, unimol_embeddings in train_loader:
            data = data.to(device)
            unimol_embeddings = unimol_embeddings.to(device)

            optimizer.zero_grad()

            output = model(data, unimol_embeddings)

            out_flat = output.view(-1)
            y_flat = data.y.view(-1)

            loss = criterion(out_flat, y_flat)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            curr_epoch = epoch + float(step) / len(train_loader)
            scheduler_warmup.step(curr_epoch)

            ema(model)
            loss_all += loss.item() * data.num_graphs
            step += 1

            if not args.no_wandb and step % 10 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/lr": optimizer.param_groups[0]['lr'],
                    "batch/step": epoch * len(train_loader) + step
                })

        train_loss = loss_all / len(train_dataset_wrapped)
        val_score, _ = evaluate(model, val_loader, ema, device, is_classification)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if compare(val_score, best_val_score):
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0

            test_score_at_best, _ = evaluate(model, test_loader, ema, device, is_classification)

            # Save best model
            save_path = osp.join(save_dir, f"fusion_{args.dataset}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'test_score': test_score_at_best,
                'config': vars(args),
                'gate_value': torch.sigmoid(model.gate_param).item()
            }, save_path)

            gate_val = torch.sigmoid(model.gate_param).item()
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.5f} | "
                  f"Val {metric_name}: {val_score:.5f} | Test {metric_name}: {test_score_at_best:.5f} | "
                  f"Gate: {gate_val:.3f} | Time: {epoch_time:.1f}s [NEW BEST]")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.5f} | "
                  f"Val {metric_name}: {val_score:.5f} | "
                  f"Patience: {patience_counter}/{args.patience} | Time: {epoch_time:.1f}s")

        if not args.no_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                f"val/{metric_name.lower()}": val_score,
                "train/lr": current_lr,
                "model/gate_value": torch.sigmoid(model.gate_param).item(),
                "model/pamnet_scale": model.pamnet_scale.item(),
                "model/unimol_scale": model.unimol_scale.item()
            }
            if test_score_at_best is not None:
                log_dict[f"test/{metric_name.lower()}"] = test_score_at_best
            wandb.log(log_dict)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Val {metric_name}: {best_val_score:.5f}")
    print(f"Test {metric_name} at Best Val: {test_score_at_best:.5f}")
    print(f"Model saved to: {osp.join(save_dir, f'fusion_{args.dataset}_best.pt')}")
    print("=" * 70 + "\n")

    if not args.no_wandb:
        wandb.log({
            "final/best_epoch": best_epoch + 1,
            f"final/best_val_{metric_name.lower()}": best_val_score,
            f"final/test_{metric_name.lower()}": test_score_at_best
        })
        wandb.finish()


if __name__ == "__main__":
    main()
