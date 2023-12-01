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
            'Bursty suppressed by contrast': 44,
            "ON bursty": 45,
            "ON delayed": 46,
            "ON small OFF large": 47,
        }

        self.cell_class_to_id = {
            # "Direction Selective": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            # "ON-OFF small RF": [21, 22, 23, 24, 25, 26],
            # "Surpressed-by-Contrast": [42, 43, 44, 45, 46, 47]
            "OFF sustained": [10, 11, 12, 13, 14, 15, 16],
            "OFF transient": [17, 18, 19, 20],
            "ON sustained": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            "ON transient": [37, 38, 39, 40, 41]
        }

        self.cell_class_to_type = {
            "Transient": [17, 18, 19, 20, 37, 38, 39, 40, 41],
            "Sustained": [10, 11, 12, 13, 14, 15, 16, 29, 30, 31, 32, 33, 34, 35, 36]
        }

        self.cell_class_to_onoff = {
            "ON": [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46],
            "OFF": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
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

    def get_id_from_cell_type(self, name: str) -> int:
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

    def get_classes(self) -> list:
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
