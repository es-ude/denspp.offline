class RGC_Cell_Names:
    """Retinal Ganglion Cell Selection"""
    def __init__(self):
        self.cell_type_to_id = {
            "ONOFF DS dorsal": 0,
            "ONOFF DS temporal": 1,
            "ONOFF DS ventral": 2,
            "ONOFF DS nasal": 3,
            "ON DS dorsonasal": 4,
            "ON DS temporal": 5,
            "ON DS ventral": 6,
            "OFF sustained EW3o": 7,
            "OFF medium sustained": 8,
            "OFF vertical OS": 9,
            "OFF horizontal OS": 10,
            "OFF sustained alpha": 11,
            "OFF sustained EW1no": 12,
            "OFF transient alpha": 13,
            "OFF transient medium RF": 14,
            "OFF transient small RF": 15,
            "F mini ON": 16,
            "Local Edge Detector": 17,
            "UHD": 18,
            "HD1": 19,
            "HD2": 20,
            "F mini OFF": 21,
            "M1": 22,
            "M2": 23,
            "PixON": 24,
            "ON alpha": 25,
            "ON horizontal OS large RF": 26,
            "ON horizontal OS small RF": 27,
            "ON vertical OS large RF": 28,
            "ON vertical OS small RF": 29,
            "M6": 30,
            "ON transient EW6t": 31,
            "ON transient medium RF": 32,
            "ON transient small RF": 33,
            "Sustained suppressed-by-contrast strong surround EW28": 34
        }

        self.cell_class_to_id = {
            "Direction Selective": [0, 1, 2, 3, 4, 5, 6],
            "OFF sustained": [7, 8, 9, 10, 11, 12],
            "OFF transient": [13, 14, 15],
            "ONOFF small RF": [16, 17, 18, 19, 20, 21],
            "ON sustained": [22, 23, 24, 25, 26, 27, 28, 29],
            "ON transient": [30, 31, 32, 33]
        }

    def get_id_from_cell_type(self, name):
        """Getting the ID from specific cell type"""
        return self.cell_type_to_id.get(name) if name in self.cell_type_to_id else -1

    def get_class_to_id(self, id):
        """Getting the class of RGC for a given ID"""
        result = ''
        for key, values in self.cell_class_to_id.items():
            if id in values:
                result = key
        return result
