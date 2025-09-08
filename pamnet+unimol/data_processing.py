import logging
from unimol_tools import UniMolRepr

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

    def data_process_pamnet(self, smiles_list):
        if self.pamnet_model is None:
            logger.warning("PAMNet model is None")
            return None
        
        # TODO: pass in the data in the format pamnet expects

    def data_process_unimol(self, smiles_list):
        unimol_output = self.unimol_model.get_repr(
            smiles_list,
            return_automatic_reprs = True
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