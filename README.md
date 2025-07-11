# End-To-End Deep (Neural) Signal Pre-Processing for Offline Transient Data Analysis (DeNSPP.offline)

This Python framework is developed for analysing transient data to evaluate an user-specific implementation of an 
end-to-end signal processor in any application. The journey of this framework starts in the research project Sp:AI:ke, 
funded by MERCUR foundation (Germany) with searching a neural signal processing pipeline for extracellular recordings 
with offline data (already recorded and available). 

In general, this framework aims to accelerate the deployment of signal processor pipelines in next-gen hardware development. 
For building the digital hardware accelerators (in C for microcontrollers and in Verilog/VHDL for FPGAs/ASICs), there is the access
to the [elasticAI.creator](https://github.com/es-ude/elastic-ai.creator).

# Table of Content
1. [Installation guide](#installation-guide)
3. [Project File structure](#project-file-structure)
4. [Data File Structure](#data-file-structure)
5. [Advanced installation guide](#advanced-installation-guide)
    - [PySpice](#pyspice)
    - [Package Manager](#uv-package-manager)
6. [Informations about third-party APIs](#more-information-about-other-apis)
7. [Citation](#citation--documentation)

## Installation guide

For using this framework, the following software tools are necessary / recommended.
- `uv` package manager ([Link](https://docs.astral.sh/uv/), [Using](https://www.saaspegasus.com/guides/uv-deep-dive/))
- Git ([Link](https://git-scm.com/downloads))
- PyCharm Community Version ([Link](https://www.jetbrains.com/de-de/pycharm/download/#section=windows))

It is recommended that each working group will create a new branch for editing their input. At the end of each project,
their input will be merged with the main branch. Further information about this software framework are described in the
paper at the end of the readme file.

If you create a new repo and you want to use the functionalities of this Python tool. Then please initialise the project.toml using `uv` and write the following code into the terminal.
```
uv add "git+https://github.com/es-ude/denspp.offline.git"
```
After initialisation, then run the following code to create the project folder structure in the Python Console.
```
from denspp.offline.structure_builder import init_project_folder
init_project_folder()
```
Important! Please start every script from your repo folder!

## Project File structure

This Git project contains the main folder `denspp` which includes the functions to setup a signal processing pipeline incl. plotting examples and training Deep Neural Networks (DNN).

The Python folder contains the following folder structure using ``init_project_folder()`` in your use-case repo:
```bash
  <path_to_project>
    ├── config              # Includes yaml files for configuring the pipelines (only local)
    ├── data                # Include transient data (only local)
    ├── dataset             # Include datasets for DNN training (only local)
    ├── runs                # Saving all results from analysis (only local)
    ├── src                 # Contains new function which can be added to the DeNSPP.offline repo (with tests)
    ├── src_dnn             # Contains functions for training deep neural networks which are not in package 
    ├── src_pipe            # Contains functions for data analysis which are not in package
    ├── access_cloud.yaml   # Yaml file with information for cloud access 
    ├── main_pipeline.py    # For normal pipeline processing 
    ├── main_data_merge.py  # For merging of given datasets
    ├── main_dnn_train.py   # For training ML models, like Autoencoder and Classifier
    ├── run_tests.py        # Function for running tests of folder content in 'src'
    └── ...
```
The marked folders with "only local" will only be saved on your computer. It not be sync with git.

## Data File Structure

In order to load transient data into the pipeline, there are two possibilities. You can share your data via:
1. OwnCloud applicatins (e.g. sciebo, ...) with defining the sharing link into the `access_cloud.yaml` in the project main folder
2. Saving the data locally by defining the path in the `Config_PipelineData.yaml` file (example later). If the API will not be find locally, then it tries to get it from remote. Otherwise an `FileNotFoundError` exception raises.

In both cases, you need a two-level file structure which is shown below. Each folder depends on the used experiment and the used hardware and data handler.
  ```bash
  <path_to_data_folder>               # Remote or local
    ├── 00_Merged                     # Includes datasets to train deep learning models
    ├── 01_<name>                     # Data folder of experiment #1 (only one level)
    │   ├── <file_01>.<format>        # Source files
    │   ├── <file_02>.<format>   
    │   ├── ... 
    │   └── Mapping_*.csv             # Electrode Mapping for loading into pipeline (opt.)
    ├── 02_<name>                     # Data folder of experiment #2 (two level)
    │   ├── <file_01>.<format>        # Experiment day #1
    │   │     ├── <file_01>.<format>  # Source files
    │   │     ├── <file_02>.<format>   
    │   │     └── ... 
    │   ├── <file_01>.<format>        # Experiment day #2
    │   │     ├── <file_01>.<format>  # Source files
    │   │     ├── <file_02>.<format>   
    │   │     └── ... 
    │   ├── ...     
    │   │     └── ... 
    │   ├── Mapping_*.csv             # Electrode Mapping for loading into pipeline (opt.)
    └── ...
```
Important is that there are two different styles for storing data. Please keep this convention otherwise no data will be loaded.
Also, please give the folder unique names for finding it in the pipeline (e.g. `01_SimData_Martinez`, then you give `martinez` into the config file and it will find it).

If you want to load a file, please change the entries in the `Config_PipelineData.yaml` file. 
- `path`: Contains the path to your local data folder  
- `data_set`: Contains the main folder with experimental data (level 0)
- `data_case`: Index for which experimental folder from `data_set` should be used (only number) (opt.)
- `data_point`: Index for which file should be used

Important, each new data structure needs a data handler for loading these data into the pipeline. Therefore please add a new function in `src_neuro/call-spike.py` or create a similar file. 

## Advanced installation guide 

### `uv` package manager
We recommended to install all python packages for using this API with a virtual environment (venv). Therefore, we also recommend to `uv` package manager. `uv` is not a standard package installed on your OS. For this, you have to install it in your Terminal (Powershell on Windows) with the following line.
````
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````
Afterwards you can create the venv and installing all packages using this line.
````
uv sync (--refresh --upgrade)
````
In the last step, you activate the Python Environment of your ``venv``. 

### PySPICE

`PySPICE` (Python Interface for using `NGSpice`) is a easy-to-use circuit simulation using SPICE commands in Python.
This framework provides several workarounds for using PySpice in order to allow easy-to-use circuit simulation for single-pole
and two-pole circuits.

#### Setup on Windows

For using PySPICE on Windows, please type in the two following commands in the Terminal (Command Prompt) after the
package is installed. We recommend to use a venv due to missing script `pyspice-post-installation` if PySPICE is
installed for all users.

Installation of the missing *.dll library in Windows:

````
pyspice-post-installation --install-ngspice-dll
````

Checking if the installation is right:

````
pyspice-post-installation --check-install
````

#### Setup on Linux

> [!TIP]
> Based on [Official Documentation](https://pyspice.fabrice-salvaire.fr/releases/v1.5/installation.html#on-linux)

To use PySpice on Linux, the ngspice package is required.
The current used PySpice implementation only supports ngspice up to version 34.
Most package managers only provide the latest version of ngspice which is far greater!

> [!NOTE] 
> **Installation on Fedora**
> ```bash
> # Enable community repo for ngspice
> dnf copr enable fabricesalvaire/ngspice
> 
> # install latest sufficient version
> dnf install libngspice-32
> ```

## More information about 3rd party tools
- `PySpice` [Official Documentation](https://pyspice.fabrice-salvaire.fr/releases/v1.5/)

## Citation / Documentation
If you want to understand what is the idea behind DeNSPP, please have a look on the corresponding [paper](https://doi.org/10.1515/cdbme-2023-1118).