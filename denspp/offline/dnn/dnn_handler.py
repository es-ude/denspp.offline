from dataclasses import dataclass
from os.path import exists, join
from denspp.offline.structure_builder import get_path_to_project_start


@dataclass
class ConfigMLPipeline:
    """Configuration class for handling the training phase of deep neural networks
    Attributes:
        mode_train_dnn:             Integer value of selected training routine regarding the training handler
        path2yaml:                  String with path to the folder with yaml configuration files
        do_plot:                    Boolean value to generate the plots after training
        do_block:                   Boolean value to block the generated plots after training
        autoencoder_mode:           Integer value for selecting the autoencoder mode [0: normal, 1: Denoising Autoencoder with mean, 2: Denoising Autoencoder with adding random noise on input, 3: Denoising Autoencoder with adding guassian noise on input]
        autoencoder_feat_size:      Integer value with dimension of the encoder output for building the feature space
        autoencoder_noise_std:      Floating value with noise std applied on autoencoder input
    """
    # --- Selection of DL Pipeline
    mode_train_dnn: int
    path2yaml: str
    # --- Options for Plotting
    do_plot: bool
    do_block: bool
    # --- Settings for Training Autoencoders
    autoencoder_mode: int
    autoencoder_feat_size: int
    autoencoder_noise_std: float

    @property
    def get_path2config(self) -> str:
        """Getting the path to the yaml config file"""
        path2start = join(get_path_to_project_start(), self.path2yaml)
        if not exists(path2start):
            raise ImportError("Folder with YAML files not available - Please check!")
        else:
            return path2start


DefaultSettings_MLPipe = ConfigMLPipeline(
    mode_train_dnn=0,
    path2yaml='config',
    do_plot=True,
    do_block=True,
    autoencoder_mode=0,
    autoencoder_feat_size=0,
    autoencoder_noise_std=0.05
)
