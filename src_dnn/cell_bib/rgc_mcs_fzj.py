from denspp.offline.data_call.call_cellbib import CellMergeClass


class resort_rgc_mcs(CellMergeClass):
    cell_type_to_id = {
        "OFF Sustained": 0,
        "OFF Transient": 1,
        "ON-OFF": 2,
        "ON Sustained": 3,
        "ON Transient": 4
    }
    cell_class_to_id = {},
    cell_class_to_type = {
        "Transient": [1, 4],
        "Sustained": [0, 3]
    }
    cell_class_to_group = {
        "OFF": [0, 1],
        "ON": [3, 4]
    }

    def __init__(self):
        pass
