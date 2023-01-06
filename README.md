# MERCUR Sp:AI:ke-Project
In this MERCUR-project Sp:Ai:ke, the software framework for developing an end-to-end pipeline for BCI's is stored.

## Getting started
The software framework will be used to optimise the spike sorting pipeline for the integration in embedded systems und in ASICs for future neural implants. For this, we propose to use PyCharm Community Version ([Link](https://www.jetbrains.com/de-de/pycharm/download/#section=windows) and the Python Interpreter 3.9/3.10. 
It is also an older version of the pipeline in MATLAB available. For this you need:
- MATLAB R2022a from The Mathworks
- Additional MATLAB Toolboxs: Fuzzy Logic, Communications, ParallelComputing 

It is recommended that each working group will create a new branch for editing their input. At the end of each project, their input will be merged with the main branch. Further informations about this software framework are described in the PPTX/PDF-Documentation.

The 2_data folder has only a subset of the data. In this repo, only a few points of the synthetic datasets are available. To get access to all datasets, get in Contact with Leo Buron.

## Installation
To use the software framework locally, please use the GitHub Desktop Software to clone the desired branch of this repository to your personal computer. A detailed description is available at this [homepage](https://www.devopszones.com/2020/08/how-to-use-github-desktop-with-gitlab.html).

## Collaborate with your team
Responsible persons:
- Analogue front-end: Andreas Erbslöh - Email: andreas.erbsloeh@uni-due.de
- Digital front-end: Leo Buron - Email: leo.buron@uni-due.de
- Neural Decoder: Zia Ur Rehman - Email: ing.zia@yahoo.com

## Usage
For starting the software framework in your branch, please start Python script "main.py". Further functions and classes are outsourced into the folder "src". For AI training, please use "main_ai.py".
