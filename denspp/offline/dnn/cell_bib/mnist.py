from denspp.offline.data_call.call_cellbib import SettingsCellSelector


class resort_mnist(SettingsCellSelector):
    original_id={
        "Zero": 0,
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
        "Six": 6,
        "Seven": 7,
        "Eight": 8,
        "Nine": 9,
    },
    original_to_reduced={
        "even": [0, 2, 4, 6, 8],
        "odd": [1, 3, 5, 7, 9]
    },
    original_to_type={
        "even": [0, 2, 4, 6, 8],
        "odd": [1, 3, 5, 7, 9]
    },
    original_to_group={
        "even": [0, 2, 4, 6, 8],
        "odd": [1, 3, 5, 7, 9]
    }
