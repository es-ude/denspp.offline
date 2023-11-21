class RGC_Cell_Names:
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
            "Direction Selective": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "OFF sustained": [10, 11, 12, 13, 14, 15, 16],
            "OFF transient": [17, 18, 19, 20],
            "ONOFF small RF": [21, 22, 23, 24, 25, 26],
            "ON sustained": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            "ON transient": [37, 38, 39, 40, 41],
            "Surpressed-by-Contrast": [42, 43, 44, 45, 46, 47]
        }

    def get_id_from_cell_type(self, name) -> int:
        """Getting the ID from specific cell type"""
        return self.cell_type_to_id.get(name) if name in self.cell_type_to_id else -1

    def get_class_to_id(self, id) -> [int, str]:
        """Getting the class of RGC for a given ID"""
        result = ''
        val = 0
        for idx, (key, values) in enumerate(self.cell_class_to_id.items()):
            if id in values:
                val = idx
                result = key
        return val, result

    def get_classes(self) -> list:
        """Getting the classes as list"""
        classes = list()
        for idx, (key, values) in enumerate(self.cell_class_to_id.items()):
            if idx == 0:
                classes.append(key)
            else:
                for class0 in classes:
                    if class0 not in key:
                        classes.append(key)
        return classes


class RGC_ONOFF_FZJ:
    """Retinal Ganglion Cell Selection from Recordings from Research Center Jülich"""
    def __init__(self):
        self.cell_type_to_id = {
            "OFF Sustained": 0,
            "OFF Transient": 1,
            "ON-OFF": 2,
            "ON Sustained": 3,
            "ON Transient": 4
        }

        self.cell_class_to_id = {
            "OFF": [0, 1],
            "ON": [3, 4]
        }

    def get_id_from_cell_type(self, name) -> int:
        """Getting the ID from specific cell type"""
        return self.cell_type_to_id.get(name) if name in self.cell_type_to_id else -1

    def get_class_to_id(self, id) -> [int, str]:
        """Getting the class of RGC for a given ID"""
        result = ''
        val = 0
        for idx, (key, values) in enumerate(self.cell_class_to_id.items()):
            if id in values:
                val = idx
                result = key
        return val, result

    def get_classes(self) -> list:
        """Getting the classes as list"""
        classes = list()
        for idx, (key, values) in enumerate(self.cell_class_to_id.items()):
            if idx == 0:
                classes.append(key)
            else:
                for class0 in classes:
                    if class0 not in key:
                        classes.append(key)
        return classes
