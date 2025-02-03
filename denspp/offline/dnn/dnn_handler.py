from dataclasses import dataclass
from os.path import exists, join
from denspp.offline.structure_builder import get_path_project_start


@dataclass
class ConfigMLPipeline:
    """Handling for training depp neural networks"""
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
        path2start = join(get_path_project_start(), self.path2yaml)
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
