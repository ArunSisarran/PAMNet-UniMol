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
    
    for data in loader:
        data = data.to(device)
        output = model(data)
        
        # Squeeze output to match target shape
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    root_path = './data/admet'
    print(f"Loading {args.dataset} from {root_path}...")
    
    train_dataset = ADMET3DDataset(root_path, args.dataset, mode='train')
    val_dataset = ADMET3DDataset(root_path, args.dataset, mode='val')
    test_dataset = ADMET3DDataset(root_path, args.dataset, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = Config(
        dataset=args.dataset, 
        dim=128, 
        n_layer=4, 
        cutoff_l=5.0, 
        cutoff_g=5.0
    )
    
    model = PAMNet(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)
    
    ema = EMA(model, decay=0.999)
    
    if args.task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = 0.0
        compare = lambda x, y: x > y
    else:
        criterion = nn.MSELoss()
        best_val_score = float('inf')
        compare = lambda x, y: x < y

    print("Start Training...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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
            
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()
            
            curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
            scheduler_warmup.step(curr_epoch)
            
            ema(model)
            loss_all += loss.item() * data.num_graphs
            step += 1
            
        train_loss = loss_all / len(train_dataset)
        val_score, metric = test(model, val_loader, ema, device, args.task == 'classification')
        
        if compare(val_score, best_val_score):
            best_val_score = val_score
            test_score, _ = test(model, test_loader, ema, device, args.task == 'classification')
            
            save_path = osp.join(args.save_dir, f"pamnet_{args.dataset}_best.pt")
            torch.save(model.state_dict(), save_path)
            
            print(f'Epoch: {epoch+1:03d}, Loss: {train_loss:.5f}, Val {metric}: {val_score:.5f}, Test {metric}: {test_score:.5f} [SAVED]')
        else:
            print(f'Epoch: {epoch+1:03d}, Loss: {train_loss:.5f}, Val {metric}: {val_score:.5f}')

if __name__ == "__main__":
    main()
