import numpy as np


class _RGC_TDB:
    """Retinal Ganglion Cell Selection from Schwartz Lab"""
    def __init__(self):
        self.cell_type_to_id = {
            "ON-OFF DS - dorsal": 0,
            "ON-OFF DS - temporal": 1,
            "ON-OFF DS - ventral": 2,
            "ON-OFF DS - nasal": 3,
            "ON-OFF DS - direction unknown": 4,
            "ON-OFF - subtype unknown": 5,
            "ON DS sustained - dorsonasal": 6,
            "ON DS sustained - temporal": 7,
            "ON DS sustained - ventral": 8,
            "ON DS sustained - direction unknown": 9,
            "OFF sustained EW3o": 10,
            "OFF medium sustained": 11,
            "OFF vertical OS - asymmetric": 12,
            "OFF horizontal OS - symmetric": 13,
            "OFF sustained alpha": 14,
            "OFF sustained EW1no": 15,
            "OFF sustained - subtype unknown": 16,
            "OFF transient alpha": 17,
            "OFF transient medium RF": 18,
            "OFF transient small RF": 19,
            "OFF transient - subtype unknown": 20,
            "F-mini-ON": 21,
            "Local edge detector": 22,
            "UHD": 23,
            "HD1": 24,
            "HD2": 25,
            "F-mini-OFF": 26,
            "M1": 27,
            "M2": 28,
            "PixON": 29,
            "ON alpha": 30,
            "ON horizontal OS large RF": 31,
            "ON horizontal OS small RF": 32,
            "ON vertical OS large RF": 33,
            "ON vertical OS small RF": 34,
            "ON OS large RF - orientation unknown": 35,
            "ON OS small RF - orientation unknown": 36,
            "M6": 37,
            "ON transient EW6t": 38,
            "ON transient medium RF": 39,
            "ON transient small RF": 40,
            "ON transient - subtype unknown": 41,
            "Sustained suppressed-by-contrast strong surround EW28": 42,
            "Sustained suppressed-by-contrast no surround EW27": 43,
            "Bursty suppressed by contrast"
            : 44,
            "ON bursty": 45,
            "ON delayed": 46,
            "ON small OFF large": 47,
            "ON sustained - subtype unknown": 48,
            "Motion sensor": 49,
            "ON DS transient": 50,
            "unknown": 51,
        }

        # Abgeglichen mit Functional classes of rgctypes.org
        self.cell_class_to_id = {
            "OFF sustained": [10, 11, 12, 13, 14, 15],
            "OFF transient": [17, 18, 19, 20],
            "ON sustained": [27, 28, 29, 30, 48],
            "ON transient": [38, 39, 40]
        }
        self.cell_class_to_type = {
            "Transient": [17, 18, 19, 20, 38, 39, 40],
            "Sustained": [10, 11, 12, 13, 14, 15, 27, 28, 29, 30, 48]
        }
        self.cell_class_to_onoff = {
            "ON": [27, 28, 29, 30, 48, 38, 39, 40],
            "OFF": [10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
        }


class _RGC_ONOFF_FZJ:
    """Retinal Ganglion Cell Selection from Recordings from Research Center Jülich"""
    def __init__(self):
        self.cell_type_to_id = {
            "OFF Sustained": 0,
            "OFF Transient": 1,
            "ON-OFF": 2,
            "ON Sustained": 3,
            "ON Transient": 4
        }
        self.cell_class_to_onoff = {
            "OFF": [0, 1],
            "ON": [3, 4]
        }
        self.cell_class_to_type = {
            "Transient": [1, 4],
            "Sustained": [0, 3]
        }


def logic_combination(true_labels: np.ndarray, pred_labels: np.ndarray, translate_dict: list) -> [np.ndarray,
                                                                                                  np.ndarray]:
    """Combination of logic for Reducing Label Classes"""
    true_labels_new = np.zeros(shape=true_labels.shape, dtype=np.uint8)
    pred_labels_new = np.zeros(shape=pred_labels.shape, dtype=np.uint8)

    for idx, cluster in enumerate(translate_dict):
        for id in cluster:
            pos = np.argwhere(true_labels == id).flatten()
            true_labels_new[pos] = idx
            pos = np.argwhere(pred_labels == id).flatten()
            pred_labels_new[pos] = idx
    return true_labels_new, pred_labels_new


class CellSelector(_RGC_ONOFF_FZJ, _RGC_TDB):
    """Cell Selection Functions"""
    cell_type_to_id: dict
    cell_class_to_id: dict
    cell_class_to_onoff: dict
    cell_class_to_type: dict
    cell_class_used: dict

    def __init__(self, sel_database: int, mode=0):
        """Sel_databased: 0=RGC FZJ, 1=RGC_TDB,
        Mode selection: 0=original, 1=Reduced specific, 2= ON/OFF, 3= Sustained/Transient"""
        # Data selection
        if sel_database == 0:
            _RGC_ONOFF_FZJ.__init__(self)
        elif sel_database == 1:
            _RGC_TDB.__init__(self)

        # Mode selection
        if mode == 0:
            self.cell_class_used = self.cell_type_to_id
        elif mode == 1:
            self.cell_class_used = self.cell_class_to_id
        elif mode == 2:
            self.cell_class_used = self.cell_class_to_onoff
        elif mode == 3:
            self.cell_class_used = self.cell_class_to_type

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
                for id in values:
                    pos = np.argwhere(cluster_id == id).flatten()
                    if pos.size != 0:
                        val[pos] = idx
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
