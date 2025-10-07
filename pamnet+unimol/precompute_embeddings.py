"""
Pre-compute UniMol embeddings for QM9 dataset
Run this ONCE before training to save time
"""

import os
import sys
import os.path as osp
import torch
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
pamnet_dir = os.path.join(os.path.dirname(current_dir), "Physics-aware-Multiplex-GNN")
sys.path.append(pamnet_dir)
sys.path.append(os.path.join(pamnet_dir, "datasets"))

from qm9_dataset import QM9
from data_processing import DataProcessing


def precompute_unimol_embeddings(dataset, data_processor, save_path, batch_size=128):
    """
    Pre-compute UniMol embeddings for entire dataset and save to disk
    
    Args:
        dataset: QM9 dataset
        data_processor: DataProcessing instance
        save_path: Path to save embeddings
        batch_size: Larger batch for faster processing (no backprop needed)
    """
    print(f"Pre-computing UniMol embeddings for {len(dataset)} molecules...")
    print(f"Processing in batches of {batch_size}...")
    
    all_embeddings = []
    failed_indices = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_end = min(i + batch_size, len(dataset))
        batch_data = [dataset[j] for j in range(i, batch_end)]
        
        smiles_list = []
        for data in batch_data:
            if hasattr(data, 'smiles'):
                smiles_list.append(data.smiles)
            else:
                smiles_list.append("C")  # Fallback
                failed_indices.append(i + len(smiles_list) - 1)
        
        with torch.no_grad():
            unimol_result = data_processor.data_process_unimol(smiles_list)
            embeddings = data_processor.extract_unimol_embeddings(unimol_result)
            
            if embeddings is not None:
                all_embeddings.append(embeddings.cpu())
            else:
                print(f"Warning: Batch {i}-{batch_end} failed, using zero vectors")
                if all_embeddings:
                    emb_dim = all_embeddings[-1].shape[-1]
                else:
                    emb_dim = 512  
                
                zero_batch = torch.zeros(len(batch_data), emb_dim)
                all_embeddings.append(zero_batch)
                failed_indices.extend(range(i, batch_end))
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save to disk
    torch.save({
        'embeddings': all_embeddings,
        'failed_indices': failed_indices,
        'num_molecules': len(dataset),
        'embedding_dim': all_embeddings.shape[-1]
    }, save_path)
    
    print(f"\n{'='*60}")
    print(f"Successfully saved embeddings to {save_path}")
    print(f"Total molecules: {len(all_embeddings)}")
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Failed molecules: {len(failed_indices)} ({100*len(failed_indices)/len(dataset):.2f}%)")
    print(f"File size: {os.path.getsize(save_path) / (1024**2):.2f} MB")
    print(f"{'='*60}\n")
    
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description='Pre-compute UniMol embeddings for QM9')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--unimol_model', type=str, default='unimolv2', help='UniMol model name')
    parser.add_argument('--unimol_size', type=str, default='84m', help='UniMol model size')
    parser.add_argument('--target', type=int, default=7, help='Target property index')
    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    set_seed(480)  
    
    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    
    print("Loading QM9 dataset...")
    path = osp.join('.', 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform())
    print(f"Loaded {len(dataset)} molecules")
    
    print("\nInitializing UniMol...")
    data_processor = DataProcessing(
        pamnet_model=None,
        unimol_model=args.unimol_model,
        unimol_model_size=args.unimol_size
    )
    
    save_dir = osp.join('.', 'data', args.dataset, 'precomputed')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, 'unimol_embeddings.pt')
    
    if os.path.exists(save_path):
        response = input(f"\nEmbeddings already exist at {save_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\nStarting pre-computation...")
    print("This will take about 20-30 minutes...\n")
    
    embeddings = precompute_unimol_embeddings(
        dataset=dataset,
        data_processor=data_processor,
        save_path=save_path,
        batch_size=args.batch_size
    )
    
    print("Verifying saved embeddings...")
    loaded = torch.load(save_path)
    print(f"✓ Loaded embeddings shape: {loaded['embeddings'].shape}")
    print(f"✓ Number of molecules: {loaded['num_molecules']}")
    print(f"✓ Embedding dimension: {loaded['embedding_dim']}")
    
    print("\n" + "="*60)
    print("Pre-computation complete!")
    print("You can now use these embeddings in training with:")
    print(f"  python simple_model.py --use_precomputed")
    print("="*60)


if __name__ == "__main__":
    main()