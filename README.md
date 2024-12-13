# End-To-End Deep Neural Signal Pre-Processing (DeNSPP)

This Python package is developed in the research project Sp:AI:ke, funded by MERCUR (Germany). This software framework
enables a AI-enhanced developing of hardware accelerator in end-to-end neural signal processing pipelines for future
neural devices (Brain-Computer Interfaces, retinal implants, ...).
Actually, this framework supports the data analysis of invasive device which requires a processing of action potentials.

## Getting started

This framework can be used for optimising your end-to-end signal processing for the integration in embedded systems und
in ASICs. For using this framework, the following software tools are necessary / recommended.

- PyCharm Community Version ([Link](https://www.jetbrains.com/de-de/pycharm/download/#section=windows))
- Python Interpreter 3.11 ([Link](https://www.python.org/downloads/release/python-3116/))
- Git ([Link](https://git-scm.com/downloads))

It is recommended that each working group will create a new branch for editing their input. At the end of each project,
their input will be merged with the main branch. Further informations about this software framework are described in the
PPTX/PDF-Documentation. To get the first start of this Framework, we recommend to create a virtual virtual environment (
venv) with installing all packages from the requirement file. This can be done using the following command line in the
terminal (Command Prompt) with activing the venv.

````
python3 -m pip install -r requirements.txt
````

## File structure

This Git project contains the main folder `3_Python` which includes the functions to setup a NSP pipeline incl. plotting examples and training Deep Neural Networks (DNN).

The Python folder contains the following folder structure:

- `config`: Includes yaml files for configuring the pipelines (only local)
- `data`: Merged datasets from your dataset library in order to train deep learning networks (only local)
- `package`: All functions in order to setup the pipeline with (i) analogue, (ii) digital, (iii) neural signal
  processing and (iv) handler for train Deep Neural Networks (like Autoencoders, Recurrent Neural Networks and
  Classifiers)
- `runs`: Temporary folder which saves the results of each run (only local)
- `test`: Temporary folder for testing code which will not be pushed (only local)
- `src_dnn`: Training handlers for different ML models
- `src_[x]`: Example for generating Pipelines and specified functions for analysing signals for your application. There
  we have pipelines for analysing EMG and neural signals from invasive microelectrode arrays

## Usage

For starting the Python framework, we recommend to configure the main folder `3_Python` as starting point in PyCharm in
order to find all references in your script. For running code, you find several Python scripts for different examples.
- `main.py`: For normal pipeline processing
- `main_data_merge.py`: For merging of given datasets
- `main_dnn_train.py`: For training ML models, like Autoencoder and Classifier

  
### Installation of virtual environment

It is recommended that all python packages should be installed using a virtual environment (venv). For this you type-in the following commands into the terminal.
````
cd 3_Python
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

````

## Advanced installation guide for PySPICE

`PySPICE` (Python Interface for using `NGSpice`) is a easy-to-use circuit simulation using SPICE commands in Python.
This API provides several workarounds for using PySpice in order to allow easy-to-use circuit simulation for single-pole
and two-pole circuits.

### Setup on Windows

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

### Setup on Linux

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

## More information about other APIs
- `PySpice` [Official Documentation](https://pyspice.fabrice-salvaire.fr/releases/v1.5/)
- `impedancefitter` [Official Documentation](https://impedancefitter.readthedocs.io/en/latest/index.html)

## Citation / Documentation
If you want to understand what is the idea behind DeNSPP, please have a look on the corresponding [paper](https://doi.org/10.1515/cdbme-2023-1118).