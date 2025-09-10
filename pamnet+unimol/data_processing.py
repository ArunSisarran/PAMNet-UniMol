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
        
        mol.Chem.AddHs(mol)
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