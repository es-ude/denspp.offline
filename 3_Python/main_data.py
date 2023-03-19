import numpy as np
from src_ai.processing_data import get_frames_from_labeled_datasets
from settings import Settings

if __name__ == "__main__":
    print("\nPreparing datasets for AI Training in end-to-end spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings_afe = Settings()
    # --- Settings of Analogue Front-End
    settings_afe.fs_ana = 50e3
    settings_afe.fs_adc = 50e3
    settings_afe.n_bit_adc = 16
    settings_afe.gain_ana = 40
    settings_afe.f_filt_ana = np.array([100, 8e3])
    settings_afe.path2data = "C:/HomeOffice/Arbeit/B_MERCUR_SpAIke/6_Daten"

    # --- Settings of Data Processing
    path2file = "src_ai/data"

    # align mode = 0: no aligning, 1: maximum, 2: minimum, 3: maximum positive slop, 4: maximum negative slope
    get_frames_from_labeled_datasets(
        path2save=path2file,
        settings_afe=settings_afe,
        data_set=2,
        align_mode=3,
        use_alldata=False,
        plot_result=False
    )

