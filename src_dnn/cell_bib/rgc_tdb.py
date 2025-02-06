from denspp.offline.data_call.call_cellbib import CellMergeClass


class resort_rgc_tdb(CellMergeClass):
    cell_type_to_id = {
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
    cell_class_to_id = {
        "OFF sustained": [10, 11, 12, 13, 14, 15],
        "OFF transient": [17, 18, 19, 20],
        "ON sustained": [27, 28, 29, 30, 48],
        "ON transient": [38, 39, 40]
    }
    cell_class_to_type = {
        "Transient": [17, 18, 19, 20, 38, 39, 40],
        "Sustained": [10, 11, 12, 13, 14, 15, 27, 28, 29, 30, 48]
    }
    cell_class_to_group = {
        "ON": [27, 28, 29, 30, 48, 38, 39, 40],
        "OFF": [10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
    }

    def __init__(self):
        pass
