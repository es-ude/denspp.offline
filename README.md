# End-To-End Deep Neural Signal Pre-Processing (DeNSPP)
This Python package is developed in the research project Sp:AI:ke, funded by MERCUR (Germany). This software framework enables a AI-enhanced developing of hardware accelerator in end-to-end neural signal processing pipelines for future neural devices (Brain-Computer Interfaces, retinal implants, ...).
Actually, this framework supports the data analysis of invasive device which requires a processing of action potentials.

## Getting started
This framework can be used for optimising your end-to-end signal processing for the integration in embedded systems und in ASICs. For using this framework, the following software tools are necessary / recommended. 
- PyCharm Community Version ([Link](https://www.jetbrains.com/de-de/pycharm/download/#section=windows))
- Python Interpreter 3.11 ([Link](https://www.python.org/downloads/release/python-3116/))
- Git ([Link](https://git-scm.com/downloads))

It is recommended that each working group will create a new branch for editing their input. At the end of each project, their input will be merged with the main branch. Further informations about this software framework are described in the PPTX/PDF-Documentation. To get the first start of this Framework, we recommend to create a virtual virtual environment (venv) with installing all packages from the requirement file. This can be done using the following command line in the terminal (Command Prompt) with activing the venv.
````
python3 -m pip install -r requirements.txt
````

## File structure
This Git project contains the three folders:
- `1_Docu`: Documentation 
- `2_Data`: Containing neural datasets. In this repo, only the synthetic dataset of Martinez2009 is available.
- `3_Python`: Containing the functions in order to setup a NSP pipeline incl. plotting examples

The Python folder contains the following folder structure:
- `data`: Merged datasets from your dataset library in order to train deep learning networks (only local)
- `experiments`: Custom-made functions for characterizing pipeline segments (e.g. spike detection)
- `package`: All functions in order to setup the pipeline with (i) analogue, (ii) digital, (iii) neural signal processing and (iv) handler for train Deep Neural Networks (like Autoencoders, Recurrent Neural Networks and Classifiers)
- `runs`: Temporary folder which saves the results of each run (only local)
- `test`: Temporary folder for testing code which will not be pushed (only local)
- `src_dnn`: Training handlers for different ML models
- `src_[x]`: Example for generating Pipelines and specified functions for analysing signals for your application. There we have pipelines for analysing EMG and neural signals from invasive microelectrode arrays
  
## Usage
For starting the Python framework, we recommend to configure the main folder `3_Python` as starting point in PyCharm in order to find all references in your script. For running code, you find several Python scripts for different examples.
- `main.py`: For normal pipeline processing
- `main_data_merge.py`: For merging of given datasets
- `main_dnn_train.py`: For training ML models, like Autoencoder and Classifier
- `main_emg.py`: For processing EMG signals
- `main_fpga.py`: For processing data using in Xilinx simulation environment and transfer the weights of a trained model in memory file

## Advanced installation guide for PySPICE
`PySPICE` (Python Interface for using `NGSpice`) is a easy-to-use circuit simulation using SPICE commands in Python. A comment of PySPICE, it has several bugs in the code but this integration has included some workarounds for easy-to-use. 

### Workaround #1: Using on Windows
For using PySPICE on Windows, please type in the two following commands in the Terminal (Command Prompt) after the package is installed. We recommend to use a venv due to missing script `pyspice-post-installation` if PySPICE is installed for all users.

Installation of the missing *.dll library in Windows:
````
pyspice-post-installation --install-ngspice-dll
````

Checking if the installation is right:
````
pyspice-post-installation --check-install
````

### Workaround #2: Transient Simulation with current controlled source and arbitrary waveforms
For using the transient simulation with current controlled source and arbitrary waveforms, here is the guideline for a workaround to fix the package bug.
1. Download the tar.gz file of NgSpice version 34 ([Link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/ngspice-34.tar.gz/download)) and unzip: 
2. In `ngspice-34\src\spicelib\devices\isrc\isrc.c` source code, edit line 30 ([based on Link](https://sourceforge.net/u/thejackal360/ngspice/ci/276a456659aa276a30140b59eff39ececdaaadb1/)):
   Old: IOP ("external", ISRC_EXTERNAL, IF_REALVEC,"external source description"),
   New: IOP ("external", ISRC_EXTERNAL, IF_STRING,"external source description"),
3. Build project "ngspice-34\visualc\sharedspice.vcxproj" in Visual Studio 2022 with the Release x64 option
   (see Section 19.1.3 of NgSpice user manual)
4. Copy "ngspice-34\visualc\sharedspice\Release.x64\ngspice.dll"
5. In PySpice package location, go to "Spice\NgSpice\Spice64_dll\dll-vs" and replace both "ngspice.dll" and "ngspice-34.dll"
   with the file copied in step 6, renaming where necessary.
