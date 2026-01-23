from denspp.player import GeneralPlayerController, DensppGenerationPlayerConfig
from pathlib import Path

if __name__ == "__main__":
    config_signal_generation=DensppGenerationPlayerConfig(logging_lvl="INFO",
                                                        input=Path("syntheticSineWave"),
                                                        target_hardware="DensPPPlayer_import",
                                                        output_open= None,
                                                        start_time= 1,
                                                        end_time= 2,
                                                        do_cut=True,
                                                        do_resample=True,
                                                        target_sampling_rate=1000,
                                                        translation_value_voltage=0.03,
                                                        channel_mapping=[0, False, 0, False])
    controller = GeneralPlayerController(class_config=config_signal_generation)


    #controller = GeneralPlayerController()