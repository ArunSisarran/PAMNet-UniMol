import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from tdc.single_pred import ADME, Tox
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("RDKit is required for ADMET 3D generation.")

class ADMET3DDataset(InMemoryDataset):
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'P': 7, 'Br': 8, 'I': 9}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, tdc_dataset_name, mode='train', transform=None, pre_transform=None, pre_filter=None):
        self.tdc_name = tdc_dataset_name
        self.mode = mode # 'train', 'val', or 'test'
        super(ADMET3DDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.tdc_name}_{self.mode}.pt']

    def download(self):
        pass

    def process(self):
        print(f"Loading {self.tdc_name} from TDC...")
        try:
            loader = ADME(name=self.tdc_name)
        except:
            try:
                loader = Tox(name=self.tdc_name)
            except:
                raise ValueError(f"Dataset {self.tdc_name} not found in ADME or Tox.")

        split = loader.get_split(method='scaffold', seed=42)
        df = split[self.mode]
        
        data_list = []
        print(f"Generating 3D graphs for {self.mode} split ({len(df)} samples)...")
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row['Drug']
            target = row['Y']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue

            mol = Chem.AddHs(mol)
            
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            if res != 0:
                res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            
            if res != 0: 
                continue
                
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass

            N = mol.GetNumAtoms()
            conf = mol.GetConformer()
            
            pos = []
            type_idx = []
            atomic_number = []
            
            valid_mol = True
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in self.types:
                    valid_mol = False # Skip exotic atoms
                    break
                type_idx.append(self.types[sym])
                atomic_number.append(atom.GetAtomicNum())
            
            if not valid_mol: continue
            
            for i in range(N):
                pos.append(list(conf.GetAtomPosition(i)))

            x = torch.tensor(type_idx, dtype=torch.long)
            z = torch.tensor(atomic_number, dtype=torch.long)
            pos = torch.tensor(pos, dtype=torch.float)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [self.bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            
            if edge_index.numel() > 0:
                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_type = edge_type[perm]

            y = torch.tensor([target], dtype=torch.float).view(1, -1)

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index, y=y)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print(f"Processed {len(data_list)} valid graphs.")
        torch.save(self.collate(data_list), self.processed_paths[0])
