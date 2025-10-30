import torch
import os.path as osp

def create_fixed_split_indices(dataset_size=130831, train_size=110000, val_size=10000, seed=480):
    torch.manual_seed(seed)
    
    # Generate random permutation
    indices = torch.randperm(dataset_size).tolist()
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Save to file
    save_path = osp.join('.', 'data', 'QM9', 'split_indices.pt')
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'seed': seed,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_indices)
    }, save_path)
    
    print(f"Split indices saved to {save_path}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return save_path

if __name__ == "__main__":
    create_fixed_split_indices()
