import os
import argparse
from datasets.admet_dataset import ADMET3DDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Caco2_Wang', help='Name of TDC dataset')
    parser.add_argument('--root', type=str, default='./data/admet', help='Data storage directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        os.makedirs(args.root)
        
    print("1. Processing Training Split...")
    train_set = ADMET3DDataset(root=args.root, tdc_dataset_name=args.dataset, mode='train')
    
    print("\n2. Processing Validation Split...")
    val_set = ADMET3DDataset(root=args.root, tdc_dataset_name=args.dataset, mode='val')
    
    print("\n3. Processing Test Split...")
    test_set = ADMET3DDataset(root=args.root, tdc_dataset_name=args.dataset, mode='test')
    
    print(f"\nDone! Data saved to {args.root}/processed/")
    print(f"Train size: {len(train_set)}")
    print(f"Val size:   {len(val_set)}")
    print(f"Test size:  {len(test_set)}")

if __name__ == "__main__":
    main()
