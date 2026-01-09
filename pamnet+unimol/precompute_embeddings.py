import os
import sys
import os.path as osp
import torch
from tqdm import tqdm
import argparse
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "datasets"))

from data_processing import DataProcessing


def precompute_unimol_embeddings(dataset, data_processor, save_path, batch_size=128):
    all_embeddings = []
    failed_indices = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_end = min(i + batch_size, len(dataset))
        batch_data = [dataset[j] for j in range(i, batch_end)]
        
        # Extract SMILES from dataset
        smiles_list = []
        for idx, data in enumerate(batch_data):
            if hasattr(data, 'smiles'):
                smiles_list.append(data.smiles)
            else:
                try:
                    from rdkit import Chem
                    mol = Chem.RWMol()
                    
                    type_to_symbol = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 
                                     5: 'S', 6: 'Cl', 7: 'P', 8: 'Br', 9: 'I'}
                    
                    for atom_type in data.x.tolist():
                        symbol = type_to_symbol.get(atom_type, 'C')
                        mol.AddAtom(Chem.Atom(symbol))
                    
                    if data.edge_index.numel() > 0:
                        added_bonds = set()
                        for j in range(data.edge_index.shape[1]):
                            begin = int(data.edge_index[0, j])
                            end = int(data.edge_index[1, j])
                            bond_key = tuple(sorted([begin, end]))
                            if bond_key not in added_bonds and begin != end:
                                mol.AddBond(begin, end, Chem.BondType.SINGLE)
                                added_bonds.add(bond_key)
                    
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                except Exception as e:
                    print(f"Warning: Failed to convert molecule {i+idx} to SMILES: {e}")
                    smiles_list.append("C")
                    failed_indices.append(i + idx)
        
        with torch.no_grad():
            unimol_result = data_processor.data_process_unimol(smiles_list)
            embeddings = data_processor.extract_unimol_embeddings(unimol_result)
            
            if embeddings is not None:
                all_embeddings.append(embeddings.cpu())
            else:
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
    
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description='Pre-compute UniMol embeddings')
    
    parser.add_argument('--dataset_type', type=str, default='QM9', 
                       choices=['QM9', 'ADMET'],
                       help='Type of dataset: QM9 or ADMET')
    parser.add_argument('--dataset_name', type=str, default='QM9',
                       help='For QM9: "QM9". For ADMET: "Caco2_Wang", "Lipophilicity_AstraZeneca", etc.')
    parser.add_argument('--qm9_root', type=str, default='./data/QM9',
                       help='Root directory for QM9 data')
    parser.add_argument('--admet_root', type=str, default='./data/admet',
                       help='Root directory for ADMET data')
    parser.add_argument('--target', type=int, default=7,
                       help='Target property index (QM9 only)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for processing')
    parser.add_argument('--unimol_model', type=str, default='unimolv2',
                       help='UniMol model name')
    parser.add_argument('--unimol_size', type=str, default='84m',
                       help='UniMol model size')
    
    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    seed = 480 if args.dataset_type == 'QM9' else 42
    set_seed(seed)
    
    print(f"PRE-COMPUTING EMBEDDINGS FOR {args.dataset_type}")
    
    data_processor = DataProcessing(
        pamnet_model=None,
        unimol_model=args.unimol_model,
        unimol_model_size=args.unimol_size
    )
    
    if args.dataset_type == 'QM9':
        from datasets import QM9
        
        class MyTransform(object):
            def __call__(self, data):
                target = args.target
                if target in [7, 8, 9, 10]:
                    target = target + 5
                data.y = data.y[:, target]
                return data
        
        print(f"Loading QM9 dataset (target={args.target})...")
        dataset = QM9(args.qm9_root, transform=MyTransform())
        print(f"Loaded {len(dataset)} molecules\n")
        
        save_dir = osp.join(args.qm9_root, 'precomputed')
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, 'unimol_embeddings.pt')
        
        if os.path.exists(save_path):
            response = input(f"Embeddings already exist at {save_path}. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        
        embeddings = precompute_unimol_embeddings(
            dataset=dataset,
            data_processor=data_processor,
            save_path=save_path,
            batch_size=args.batch_size
        )
        
        loaded = torch.load(save_path)
        
    # ADMET PROCESSING
    elif args.dataset_type == 'ADMET':
        from datasets.admet_dataset import ADMET3DDataset
        
        print(f"Loading {args.dataset_name} dataset...")
        
        train_dataset = ADMET3DDataset(args.admet_root, args.dataset_name, mode='train')
        val_dataset = ADMET3DDataset(args.admet_root, args.dataset_name, mode='val')
        test_dataset = ADMET3DDataset(args.admet_root, args.dataset_name, mode='test')
        
        save_dir = osp.join(args.admet_root, 'processed', args.dataset_name, 'precomputed')
        os.makedirs(save_dir, exist_ok=True)
        
        for split_name, dataset in [('train', train_dataset), 
                                     ('val', val_dataset), 
                                     ('test', test_dataset)]:
            
            save_path = osp.join(save_dir, f'unimol_embeddings_{split_name}.pt')
            
            if os.path.exists(save_path):
                print(f"\n{split_name} embeddings already exist at {save_path}")
                response = input(f"Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print(f"Skipping {split_name}...\n")
                    continue
            
            embeddings = precompute_unimol_embeddings(
                dataset=dataset,
                data_processor=data_processor,
                save_path=save_path,
                batch_size=args.batch_size
            )
        
if __name__ == "__main__":
    main()
