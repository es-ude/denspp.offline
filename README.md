# End-To-End Deep Neural Signal Pre-Processing (DeNSSP)
This Python package is developed in the research project Sp:AI:ke, funded by MERCUR (Germany). This software framework enables a AI-enhanced developing of hardware accelerator in end-to-end neural signal processing pipelines for future neural devices (Brain-Computer Interfaces, retinal implants, ...).
Actually, this framework supports the data analysis of invasive device which requires a processing of action potentials.

## Getting started
This framework can be used for optimising your end-to-end signal processing for the integration in embedded systems und in ASICs. For using this framework, the following software tools are necessary / recommended.
- PyCharm Community Version ([Link](https://www.jetbrains.com/de-de/pycharm/download/#section=windows))
- Python Interpreter 3.11 ([Link](https://www.python.org/downloads/release/python-3116/))
- Git ([Link](https://git-scm.com/downloads))

It is recommended that each working group will create a new branch for editing their input. At the end of each project, their input will be merged with the main branch. Further informations about this software framework are described in the PPTX/PDF-Documentation.

The folder "2_Data" contains only a subset of the data. In this repo, only a few points of the synthetic datasets are available.

## File structure
This Git project contains the three folders:
- _1_Docu_: Documentation 
- _2_Data_: Containing neural datasets. In this repo, only the synthetic dataset of Martinez2009 is available.
- _3_Python_: Containing the functions in order to setup a NSP pipeline incl. plotting examples

The Python folder contains the following folder structure:
- _data_: Merged datasets from your dataset library in order to train deep learning networks
- _experiments_: Custom-made functions for characterizing pipeline segments (e.g. spike detection)
- _package_: All functions in order to setup the pipeline with (i) analogue, (ii) digital, (iii) neural signal processing and (iv) handler for train Autoencoders
- _runs_: Temporary folder which saves the results of each run
- _src_decoder_: Pipeline and necessary functions decoding informations from recordings with Utah electrode arrays
- _src_emg_: Pipeline and necessary functions for analysing EMG signals
- _src_neuro_: Pipeline and necesary functions for analysing action potentials from invasive microelectrode arrays


## Collaborate with your team
Responsible persons:
- *Analogue front-end*: Andreas Erbslöh - Email: andreas.erbsloeh@uni-due.de
- *Digital front-end*: Leo Buron - Email: leo.buron@uni-due.de
- *Neural Decoder*: Zia Ur Rehman - Email: ing.zia@yahoo.com

## Usage
For starting the software framework in your branch, please start Python script ...
- "main.py" for normal pipeline processing
- "main_ai_autoencoder.py" for AI training of the deep learning structure of Autoencoders
- "main_data.py" for merging of given datasets
- "main_decoder.py" for decoding data from Utah electrode arrays
- "main_fpga.py" for processing data using in Xilinx simulation environment and transfer the weights of a trained model in memory file 
