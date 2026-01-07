import numpy as np
from dataclasses import dataclass
from denspp.offline import check_key_elements, check_elem_unique


@dataclass
class SettingsCellSelector:
    """Class for Merging different labels of a dataset to a new dataset
    Attributes:
         original_id:           Dictionary with original label names [key] and given id [value]
         original_to_reduced:   Dictionary with new label class [key] and corresponding id as list [value]
         original_to_group:     Dictionary with new label subclass [key] and corresponding id as list [value]
         original_to_type:      Dictionary with new label subclass [key] and corresponding id as list [value]
    """
    original_id: dict
    original_to_reduced: dict
    original_to_group: dict
    original_to_type: dict


class CellSelector:
    _used_id_library: dict
    _process_mode: bool

    def __init__(self, cell_merge: SettingsCellSelector, mode: int) -> None:
        """Class for separating the classes into new subset
        :param cell_merge: Class with handling process
        :param mode:       0=original dataset, 1=reduced dataset, 2= original to subgroup dataset, 3= original to subtype dataset
        :return:           None
        """
        # Mode selection
        assert mode in [0, 1, 2, 3], "Wrong mode selected - Please check!"
        self._data_origin = mode == 0
        if mode == 0:
            self._used_id_library = cell_merge.original_id
        elif mode == 1:
            self._used_id_library = cell_merge.original_to_reduced
        elif mode == 2:
            self._used_id_library = cell_merge.original_to_group
        elif mode == 3:
            self._used_id_library = cell_merge.original_to_type

    def get_id_from_key(self, name: str) -> int | list:
        """Getting the ID from a cell type / class name / key
        :param name:    Key name of the label from dataset
        :return:        Corresponding ID to key label
        """
        keylist = [key for key in self._used_id_library.keys()]
        assert check_key_elements(name, keylist), f"Key not available: {keylist}"
        return self._used_id_library.get(name)

    def get_name_from_id(self, cluster_id: int | np.ndarray) -> str:
        """Getting the name of the cell type of a given cluster ID/class
        :param cluster_id:  Cluster ID
        :return:            String with label
        """
        if self._data_origin:
            cell_name = [key for key, values in self._used_id_library.items() if cluster_id == values]
        else:
            cell_name = [key for key, values in self._used_id_library.items() if cluster_id in values]
        return cell_name[0] if len(cell_name) else ''

    def get_label_list(self) -> list:
        """Getting the label names of used dataset as list
        :return:    List of used cell type names as label
        """
        keylist = [key for key in self._used_id_library.keys()]
        assert check_elem_unique(keylist), f"Keys of dataset labels are not unique - Please check!"
        return keylist

    def transform_label_to_id_integer(self, old_id: int) -> int:
        """Function for transforming the old ID to the new ID using the translation dictionary
        :param old_id:  Integer with old ID from original dataset
        :return:        Integer with new ID for new dataset (values with -1 are not defined and must be removed)
        """
        if self._data_origin:
            cell_name = [values for values in self._used_id_library.values() if old_id == values]
        else:
            cell_name = [idx for idx, values in enumerate(self._used_id_library.values()) if old_id in values]
        return cell_name[0] if len(cell_name) else -1

    def transform_label_to_id_array(self, old_id: np.ndarray) -> np.ndarray:
        """Function for transforming the old ID to the new ID using the translation dictionary
        :param old_id:  Numpy array with old IDs from original dataset
        :return:        Numpy array with new IDs for new dataset (values with -1 are not defined and must be removed)
        """
        new_label = np.zeros_like(old_id)
        for idx, value in enumerate(old_id):
            new_label[idx] = self.transform_label_to_id_integer(value)
        return new_label

    def transform_data_into_new(self, old_id: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function for transforming the old ID and data to the new format
        :param old_id:  Numpy array with old IDs from original dataset
        :param data:    Numpy array with dataset for training with shape (num of samples, num of features)
        :return:        Numpy array with new (0) IDs and (1) data for new dataset
        """
        if self._data_origin:
            return old_id, data
        else:
            new_id = self.transform_label_to_id_array(old_id)
            xpos_sel = np.argwhere(new_id != -1).flatten()
            return new_id[xpos_sel], data[xpos_sel, :]
