# First steps

Welcome to the denspp.offline Framework. This Python framework can be used for anaylsing transient data from any application / use-case.

## Usable Techniques
- Analysis of transient data for evaluating a real-time suitable signal processing pipeline
- Applying pre-processing methods on transient data to achieve an event-based signal processing
- Generating datasets for Machine-Learning- (ML) or Deep-Neural-Network-based (DNN) models
- Training of ML-based and DNN-based models incl. report generation
- Transferring the best methods into an embedded system (MCU, FPGA) or ASIC design

## Further Information
If you need information for installing and initialising the framework, please check the [main GitHub page](https://es-ude.github.io/denspp.offline) or the [readme](https://github.com/es-ude/denspp.offline) file.
If you want to know which methods are available, please check the API reference on the [Github pages](https://es-ude.github.io/denspp.offline/apidocs/index.html).

## Installation / First use
1. We recommend to install the *uv* package manager from [here](https://docs.astral.sh/uv/getting-started/installation/)
2. Creating a new Python Project Environment, e.g. using the PyCharm Community Edition ([Download](https://www.jetbrains.com/de-de/pycharm/download/other.html))
3. Using *uv* in the terminal with ``uv sync`` for building the virtual environment and installing the necessary Python version
4. After the initialisation phase, please run the following two lines in the Python Console from your project start folder to generate the project structure. The result should be similar to the example in the readme 
   1. ``from denspp.offline.structure_builder import init_project_folder``
   2. ``init_project_folder()``
5. Run the main files in order to build necessary yaml files. For example, you can upload your files to a cloud storage like OwnCloud. Therefore, you have to establish the folder structure described on the GitHub pages or in the readme. After that, you have to add the calls in the ``call_data.py`` file for the transient data or in ``call_datasets.py`` file for datasets.
6. Using files can be configured using the use-case specific yaml file. These files will be downloaded from the cloud storage (if they are not locally). 
7. Enjoy the examples and adapt them for your application!
8. If you want to update all repos in your venv, please use the command ``uv sync --refresh --upgrade`` in the terminal.
