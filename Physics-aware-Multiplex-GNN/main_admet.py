import os
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

from models import PAMNet, Config
from datasets.admet_dataset import ADMET3DDataset
from utils import EMA

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(model, loader, ema, device, is_binary):
    model.eval()
    ema.assign(model)
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            
            pred = output.view(-1)
            target = data.y.view(-1)
            
            if is_binary:
                pred = torch.sigmoid(pred)
                
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
        
    ema.resume(model)
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    
    if is_binary:
        try:
            return roc_auc_score(targets, preds), "AUC"
        except:
            return 0.5, "AUC"
    else:
        return mean_absolute_error(targets, preds), "MAE"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Caco2_Wang')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'])
    parser.add_argument('--epochs', type=int, default=300)  
    parser.add_argument('--lr', type=float, default=5e-4)  
    parser.add_argument('--wd', type=float, default=1e-5)  
    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--dim', type=int, default=128)  
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--patience', type=int, default=50)  
    parser.add_argument('--dropout', type=float, default=0.1)  
    parser.add_argument('--grad_clip', type=float, default=5.0)  
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.995)  
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    root_path = './data/admet'
    
    train_dataset = ADMET3DDataset(root_path, args.dataset, mode='train')
    val_dataset = ADMET3DDataset(root_path, args.dataset, mode='val')
    test_dataset = ADMET3DDataset(root_path, args.dataset, mode='test')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    config = Config(
        dataset=args.dataset, 
        dim=args.dim, 
        n_layer=args.n_layer, 
        cutoff_l=5.0, 
        cutoff_g=5.0
    )
    
    model = PAMNet(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer, 
        multiplier=1.0, 
        total_epoch=args.warmup_epochs, 
        after_scheduler=scheduler
    )
    
    ema = EMA(model, decay=0.999)
    
    if args.task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = 0.0
        compare = lambda x, y: x > y
        mode_str = "max"
    else:
        criterion = nn.L1Loss()  
        best_val_score = float('inf')
        compare = lambda x, y: x < y
        mode_str = "min"
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    best_epoch = 0
    patience_counter = 0
    test_score_at_best = None
    
    for epoch in range(args.epochs):
        model.train()
        loss_all = 0
        step = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            
            out_flat = output.view(-1)
            y_flat = data.y.view(-1)
            
            loss = criterion(out_flat, y_flat)
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            optimizer.step()
            
            curr_epoch = epoch + float(step) / len(train_loader)
            scheduler_warmup.step(curr_epoch)
            
            ema(model)
            loss_all += loss.item() * data.num_graphs
            step += 1
            
        train_loss = loss_all / len(train_dataset)
        val_score, metric = test(model, val_loader, ema, device, args.task == 'classification')
        current_lr = optimizer.param_groups[0]['lr']
        
        if compare(val_score, best_val_score):
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            
            test_score_at_best, _ = test(model, test_loader, ema, device, args.task == 'classification')
            
            save_path = osp.join(args.save_dir, f"pamnet_{args.dataset}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'test_score': test_score_at_best,
                'config': vars(args),
            }, save_path)
            
            print(f'Epoch {epoch+1:03d} Loss: {train_loss:.5f} '
                  f'Val {metric}: {val_score:.5f} Test {metric}: {test_score_at_best:.5f}')
        else:
            patience_counter += 1
            print(f'Epoch {epoch+1:03d} Loss: {train_loss:.5f} '
                  f'Val {metric}: {val_score:.5f} '
                  f'Patience: {patience_counter}/{args.patience}')
        
        if patience_counter >= args.patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best epoch: {best_epoch+1}")
            print(f"Best Val {metric}: {best_val_score:.5f}")
            print(f"Test {metric} at best: {test_score_at_best:.5f}")
            print(f"{'='*70}\n")
            break
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation {metric}: {best_val_score:.5f} (Epoch {best_epoch+1})")
    print(f"Test {metric} at Best Val: {test_score_at_best:.5f}")
    print(f"Model saved to: {osp.join(args.save_dir, f'pamnet_{args.dataset}_best.pt')}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
