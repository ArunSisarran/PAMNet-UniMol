import logging
import sys
import os
from torch_geometric.data import Batch
from unimol_tools import UniMolRepr
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import numpy as np
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
pamnet_dir = os.path.join(os.path.dirname(current_dir), "Physics-aware-Multiplex-GNN")
sys.path.append(pamnet_dir)

from models import PAMNet, Config

logger = logging.getLogger(__name__)

class DataProcessing:
    """
    Passes the SMILES input to both the PAMNet model and the UniMol model
    """

    def __init__(self, pamnet_model=None, pamnet_state_dict_path="../Physics-aware-Multiplex-GNN/save/pamnet_rna.pt", 
                 unimol_model: str="unimolv2", unimol_model_size: str="84m"):

        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        if pamnet_model is None:
            try:
                state_dict = torch.load(pamnet_state_dict_path, map_location=map_location)
                
                config = Config(
                    dataset="rna",
                    dim=16,
                    n_layer=1,
                    cutoff_l=5.0,
                    cutoff_g=10.0,
                    flow='source_to_target'
                )
                
                self.pamnet_model = PAMNet(config)
                self.pamnet_model.load_state_dict(state_dict)
                self.pamnet_model.eval()
                
                logger.info(f"loaded PAMNet model from {pamnet_state_dict_path}")
                
            except Exception as e:
                logger.error(f"failed to load PAMNet model from {pamnet_state_dict_path}: {e}")
                self.pamnet_model = None
        else:
            self.pamnet_model = pamnet_model

        self.unimol_model = UniMolRepr(
            data_type="molecule",
            model_name=unimol_model,
            model_size=unimol_model_size,
            remove_hs=False
        )

    def smiles_to_pamnet(self, smiles):
        """Convert SMILES to PAMNet-compatible graph data"""
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return None
        
        mol = Chem.AddHs(mol)
        
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            logger.warning(f"Could not generate 3D coordinates for {smiles}: {e}")
        
        atomic_nums = []
        positions = []
        
        conformer = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            
            if hasattr(self, 'pamnet_model') and self.pamnet_model is not None:
                if hasattr(self.pamnet_model, 'dataset') and self.pamnet_model.dataset[:3].lower() == "rna":
                    if atomic_num == 6:  # Carbon
                        atomic_nums.append(0)
                    elif atomic_num == 7:  # Nitrogen
                        atomic_nums.append(1)
                    elif atomic_num == 8:  # Oxygen
                        atomic_nums.append(2)
                    else:
                        continue
                else:
                    atomic_nums.append(atomic_num)
            else:
                atomic_nums.append(atomic_num)
            
            pos = conformer.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        
        if not atomic_nums:
            logger.warning(f"No valid atoms found for SMILES: {smiles}")
            return None
        
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            if i >= len(atomic_nums) or j >= len(atomic_nums):
                continue
                
            bond_type = bond.GetBondTypeAsDouble()
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_type, bond_type])

        if hasattr(self, 'pamnet_model') and self.pamnet_model is not None:
            if hasattr(self.pamnet_model, 'dataset') and self.pamnet_model.dataset[:3].lower() == "rna":
                x_data = []
                for i, (pos, atom_type) in enumerate(zip(positions, atomic_nums)):
                    x_data.append(pos + [atom_type])
                x = torch.tensor(x_data, dtype=torch.float32)
            else:
                x = torch.tensor(atomic_nums, dtype=torch.long)
        else:
            x = torch.tensor(atomic_nums, dtype=torch.long)

        pos = torch.tensor(positions, dtype=torch.float32)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    def data_process_pamnet(self, smiles_list):
        """Process SMILES list through PAMNet model"""
        if self.pamnet_model is None:
            logger.warning("PAMNet model is None")
            return None

        data_list = []
        for smiles in smiles_list:
            data = self.smiles_to_pamnet(smiles)
            if data is not None:
                data_list.append(data)

        if not data_list:
            return None

        batch = Batch.from_data_list(data_list)
        
        device = next(self.pamnet_model.parameters()).device
        batch = batch.to(device)

        with torch.no_grad():
            pamnet_output = self.pamnet_model(batch)

        return pamnet_output
    
    def clean_smiles(self, smiles):
        """
        Clean SMILES for UniMol compatibility
        """
    
        if '[H]' in smiles:
            cleaned = re.sub(r'\[H\]', '', smiles)
            cleaned = re.sub(r'\(\)', '', cleaned)  
        else:
            cleaned = smiles
    
        try:
            mol = Chem.MolFromSmiles(cleaned)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                return canonical
            else:
                logger.warning(f"Invalid SMILES (RDKit can't parse): {smiles[:80]}")
                return None
        except Exception as e:
            logger.warning(f"Error processing SMILES: {smiles[:80]} - {e}")
            return None

    def data_process_unimol(self, smiles_list):
        """
        Process SMILES through UniMol with robust error handling
        """
        # Clean all SMILES
        cleaned_smiles = []
        valid_indices = []
    
        for idx, smiles in enumerate(smiles_list):
            clean = self.clean_smiles(smiles)
            if clean is not None:
                cleaned_smiles.append(clean)
                valid_indices.append(idx)
            else:
                cleaned_smiles.append("C")  
    
        try:
            unimol_output = self.unimol_model.get_repr(
                cleaned_smiles,
                return_atomic_reprs=True
            )

            if 'cls_repr' in unimol_output:
                cls_reprs = unimol_output['cls_repr']
                for idx in range(len(smiles_list)):
                    if idx not in valid_indices:
                        cls_reprs[idx] = None
                unimol_output['cls_repr'] = cls_reprs
                unimol_output['valid_indices'] = valid_indices

            return unimol_output

        except Exception as e:
            logger.error(f"UniMol processing failed: {e}")
            return None

    def extract_unimol_embeddings(self, unimol_result):
        """
        Extract embeddings, using zero vectors for invalid molecules
        """
        import numpy as np
    
        if unimol_result is None:
            return None
    
        if isinstance(unimol_result, dict) and 'cls_repr' in unimol_result:
            cls_repr = unimol_result['cls_repr']

            if isinstance(cls_repr, list):
                emb_dim = None
                for emb in cls_repr:
                    if emb is not None:
                        emb_dim = len(emb) if isinstance(emb, (list, np.ndarray)) else emb.shape[-1]
                        break

                if emb_dim is None:
                    emb_dim = 512  

                processed_reprs = []
                for emb in cls_repr:
                    if emb is not None:
                        processed_reprs.append(emb)
                    else:
                        processed_reprs.append(np.zeros(emb_dim, dtype=np.float32))

                embeddings = np.stack(processed_reprs)
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

                if embeddings_tensor.dim() == 1:
                    embeddings_tensor = embeddings_tensor.unsqueeze(0)
                elif embeddings_tensor.dim() > 2:
                    batch_size = embeddings_tensor.shape[0]
                    feature_dim = embeddings_tensor.shape[-1]
                    embeddings_tensor = embeddings_tensor.view(batch_size, feature_dim)

                return embeddings_tensor
            else:
                if not isinstance(cls_repr, torch.Tensor):
                    cls_repr = torch.tensor(cls_repr, dtype=torch.float32)

                if cls_repr.dim() == 1:
                    cls_repr = cls_repr.unsqueeze(0)
                elif cls_repr.dim() > 2:
                    batch_size = cls_repr.shape[0]
                    feature_dim = cls_repr.shape[-1]
                    cls_repr = cls_repr.view(batch_size, feature_dim)

                return cls_repr
    
        return None

    def data_process_pamnet_unimol(self, smiles_list):
        """Process SMILES through both models"""
        pamnet_result = self.data_process_pamnet(smiles_list)
        unimol_result = self.data_process_unimol(smiles_list)

        result = {
            "input_smiles": smiles_list,
            "pamnet_result": pamnet_result,
            "unimol_result": unimol_result,
            "unimol_embeddings": self.extract_unimol_embeddings(unimol_result)
        }

        return result

    def get_output_dimensions(self, sample_smiles=["CCO"]):
        """
        Helper method to determine output dimensions
        """
        results = self.data_process_pamnet_unimol(sample_smiles)
    
        dimensions = {}
    
        if results["pamnet_result"] is not None:
            dimensions["pamnet"] = results["pamnet_result"].shape[-1]
        else:
            dimensions["pamnet"] = None
        
        if results["unimol_embeddings"] is not None:
            dimensions["unimol"] = results["unimol_embeddings"].shape[-1]
        else:
            dimensions["unimol"] = None
            
        return dimensions



           
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    data_processor = DataProcessing()
    
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
    ]
    
    print(f"\nTesting DataProcessing class...")
    print(f"Test SMILES: {test_smiles}")
    
    try:
        print("\n1. Testing SMILES to PAMNet data conversion...")
        for i, smiles in enumerate(test_smiles):
            print(f"  Converting SMILES {i+1}: {smiles}")
            pamnet_data = data_processor.smiles_to_pamnet(smiles)
            if pamnet_data is not None:
                print(f"    Success! Atoms: {pamnet_data.x.shape[0]}, Edges: {pamnet_data.edge_index.shape[1]}")
                print(f"    Atomic numbers: {pamnet_data.x[:10].tolist()}...")  
                print(f"    Position shape: {pamnet_data.pos.shape}")
                if hasattr(pamnet_data, 'edge_attr'):
                    print(f"    Edge attributes shape: {pamnet_data.edge_attr.shape}")
            else:
                print(f"    Failed to convert SMILES: {smiles}")
        
        if data_processor.pamnet_model is not None:
            print("\n2. Testing PAMNet processing...")
            pamnet_result = data_processor.data_process_pamnet(test_smiles[:1])
            if pamnet_result is not None:
                print(f"    PAMNet result shape: {pamnet_result.shape}")
                print(f"    PAMNet result type: {type(pamnet_result)}")
            else:
                print("    PAMNet processing failed")
        else:
            print("\n2. Skipping PAMNet testing - model not loaded")
        
        print("\n3. Testing UniMol processing...")
        unimol_result = data_processor.data_process_unimol(test_smiles)
        if unimol_result is not None:
            print(f"    UniMol result type: {type(unimol_result)}")
            if hasattr(unimol_result, 'shape'):
                print(f"    UniMol result shape: {unimol_result.shape}")
            elif isinstance(unimol_result, dict):
                print(f"    UniMol result keys: {list(unimol_result.keys())}")
                for key, value in unimol_result.items():
                    if hasattr(value, 'shape'):
                        print(f"      {key} shape: {value.shape}")
            else:
                print(f"    UniMol result: {unimol_result}")
        
        print("\n4. Testing combined processing...")
        combined_result = data_processor.data_process_pamnet_unimol(test_smiles)
        print(f"    Combined result keys: {list(combined_result.keys())}")
        print(f"    PAMNet result: {combined_result['pamnet_result']}")
        print(f"    UniMol result type: {type(combined_result['unimol_result'])}")
        
        print("\n5. Getting output dimensions...")
        dimensions = data_processor.get_output_dimensions(test_smiles[:1])
        print(f"    Output dimensions: {dimensions}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting completed!")
