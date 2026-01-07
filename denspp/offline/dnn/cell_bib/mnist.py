from denspp.offline.data_call.call_cellbib import SettingsCellSelector


class resort_mnist(SettingsCellSelector):
    original_id=dict(
        zero=0,
        one=1,
        two=2,
        three=3,
        four=4,
        five=5,
        six=6,
        seven=7,
        eight=8,
        nine=9
    )
    original_to_reduced=dict(
        even=[0, 2, 4, 6, 8],
        odd=[1, 3, 5, 7, 9]
    )
    original_to_type=dict(
        even=[0, 2, 4, 6, 8],
        odd=[1, 3, 5, 7, 9]
    )
    original_to_group=dict(
        even=[0, 2, 4, 6, 8],
        odd=[1, 3, 5, 7, 9]
    )

    def __init__(self):
        pass
