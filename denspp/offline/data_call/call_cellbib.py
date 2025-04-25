import numpy as np
from dataclasses import dataclass


def logic_combination(true_labels: np.ndarray, pred_labels: np.ndarray, translate_dict: list) -> [np.ndarray, np.ndarray]:
    """Combination of logic for Reducing Label Classes
    :param true_labels:     Numpy array with true labels
    :param pred_labels:     Numpy array with predicted labels
    :param translate_dict:  Dictionary with label ids to combine
    :returns:               Two numpy arrays with true_labels_new and pred_labels_new
    """
    true_labels_new = np.zeros(shape=true_labels.shape, dtype=np.uint8)
    pred_labels_new = np.zeros(shape=pred_labels.shape, dtype=np.uint8)

    for idx, cluster in enumerate(translate_dict):
        for id in cluster:
            pos = np.argwhere(true_labels == id).flatten()
            true_labels_new[pos] = idx
            pos = np.argwhere(pred_labels == id).flatten()
            pred_labels_new[pos] = idx
    return true_labels_new, pred_labels_new


@dataclass
class CellMergeClass:
    """Class for Merging different labels of a dataset to a new dataset
    Attributes:
         cell_type_to_id:       Dictionary with original label names [key] and given id [value]
         cell_class_to_id:      Dictionary with new label class [key] and corresponding id as list [value]
         cell_class_to_group:   Dictionary with new label sub-class [key] and corresponding id as list [value]
         cell_class_to_type:    Dictionary with new label sub-class [key] and corresponding id as list [value]
    """
    cell_type_to_id: dict
    cell_class_to_id: dict
    cell_class_to_group: dict
    cell_class_to_type: dict


class CellSelector:
    handler: CellMergeClass
    cell_type_to_id: dict
    cell_class_to_id: dict
    cell_class_to_group: dict
    cell_class_to_type: dict
    cell_class_used: dict

    def __init__(self, cell_merge: CellMergeClass, mode: int) -> None:
        """Class for separating the classes into new subset
        Args:
            cell_merge: Class with handling process
            mode:       0=original, 1=Reduced specific, 2= ON/OFF, 3= Sustained/Transient
        Returns:
            None
        """
        # Mode selection
        if mode == 0:
            self.cell_class_used = cell_merge.cell_type_to_id
        elif mode == 1:
            self.cell_class_used = cell_merge.cell_class_to_id
        elif mode == 2:
            self.cell_class_used = cell_merge.cell_class_to_group
        elif mode == 3:
            self.cell_class_used = cell_merge.cell_class_to_type

    def get_id_from_celltype(self, name: str) -> int:
        """Getting the ID from a cell type"""
        return self.cell_type_to_id.get(name) if name in self.cell_type_to_id else -1

    def get_class_to_id(self, cluster_id: int | np.ndarray) -> int | np.ndarray:
        """Getting the class for a given ID"""
        default_value = -1
        if isinstance(cluster_id, int):
            val = default_value
            for idx, (_, values) in enumerate(self.cell_class_used.items()):
                if cluster_id in values:
                    val = idx
                    break
            return val
        else:
            val = np.zeros(shape=cluster_id.shape, dtype=np.int16) + default_value
            for idx, (_, values) in enumerate(self.cell_class_used.items()):
                if isinstance(values, list):
                    for id in values:
                        pos = np.argwhere(cluster_id == id).flatten()
                        if pos.size != 0:
                            val[pos] = idx
                else:
                    val = cluster_id
            return val

    def get_celltype_name_from_id(self, cluster_id: int | np.ndarray) -> str:
        """Getting the name of the cell type of a given cluster ID/class"""
        cell_name = ''
        for idx, (key, values) in enumerate(self.cell_class_used.items()):
            if cluster_id == values:
                cell_name = key
                break
        return cell_name

    def get_celltype_names(self) -> list:
        """Getting the classes as list"""
        classes = list()
        for idx, (key, _) in enumerate(self.cell_class_used.items()):
            if idx == 0:
                classes.append(key)
            else:
                for class0 in classes:
                    if class0 not in key:
                        classes.append(key)
                        break
        return classes
