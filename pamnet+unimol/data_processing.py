import logging
from torch_geometric.data import Batch
from unimol_tools import UniMolRepr
from rdkit import Chem
from rdkit import AllChem
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class DataProcessing:
    """
    passes the SMILES input to both the PAMNet model and the UniMol model
    """

    def __init__(self, pamnet_model=None, unimol_model: str="unimolv2", unimol_model_size: str="84m"):
        self.pamnet_model = pamnet_model
        self.unimol_model = UniMolRepr(
            data_type = "molecule",
            model_name = unimol_model,
            model_size = unimol_model_size,
            remove_hs = False
        )

    def smiles_to_pamnet(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        
        atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        conformer = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])

        x = torch.tensor(atomic_nums, dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, pos=pos)

    def data_process_pamnet(self, smiles_list):
        if self.pamnet_model is None:
            logger.warning("PAMNet model is None")
            return None

        data_list = []
        for smiles in smiles_list:
            data = self.smiles_to_pamnet_data(smiles)
            if data is not None:
                data_list.append(data)

        if not data_list:
            logger.warning("No valid molecules for PAMNet")
            return None

        batch = Batch.from_data_list(data_list)

        with torch.no_grad():
            pamnet_output = self.pamnet_model(batch)

        return pamnet_output

    def data_process_unimol(self, smiles_list):
        unimol_output = self.unimol_model.get_repr(
            smiles_list,
            return_atomic_reprs = True
        )

        return unimol_output

    def data_process_pamnet_unimol(self, smiles_list):
        pamnet_result = self.data_process_pamnet(smiles_list)
        unimol_result = self.data_process_unimol(smiles_list)

        result = {
            "input smiles": smiles_list,
            "pamnet result": pamnet_result,
            "unimol result": unimol_result
        }

        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
    ]
    
    print("Testing DataProcessing class...")
    print(f"Test SMILES: {test_smiles}")
    
    print("\n1. Testing UniMol processing only...")
    try:
        data_processor = DataProcessing()
        
        print("\n2. Testing SMILES to PAMNet data conversion...")
        for i, smiles in enumerate(test_smiles):
            print(f"  Converting SMILES {i+1}: {smiles}")
            pamnet_data = data_processor.smiles_to_pamnet(smiles)
            if pamnet_data is not None:
                print(f"    Success! Atoms: {pamnet_data.x.shape[0]}, Edges: {pamnet_data.edge_index.shape[1]}")
                print(f"    Atomic numbers: {pamnet_data.x[:10].tolist()}...")  
                print(f"    Position shape: {pamnet_data.pos.shape}")
            else:
                print(f"    Failed to convert SMILES: {smiles}")
        
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
        
        print("\n4. Testing combined processing (UniMol only)...")
        combined_result = data_processor.data_process_pamnet_unimol(test_smiles)
        print(f"    Combined result keys: {list(combined_result.keys())}")
        print(f"    PAMNet result: {combined_result['pamnet result']}")
        print(f"    UniMol result type: {type(combined_result['unimol result'])}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting completed!")