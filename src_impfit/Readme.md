# Steps before running the code
## Install the required software
Move one directory up
`pip install -r requirements.txt`
## Process the transients
Add the data by downloading from:
https://uni-duisburg-essen.sciebo.de/s/cOdCVvg2zlHiOpa/download
Unzip the folder.
The data structure is:
* folder `ReviewJNE_ImpedanceData/TransientMeasurement` that contains all transients
* the folder `impedance_expected` contains data for the MEA that we used. Replace it if you are interested in your own electrode.
# Workflow
1. Run `run_mode_sweep.py` to sweep over sampling frequency and LSB voltage
1. Run `do_statistic_run.py` to get an overview of the parameters fitted to the FFT spectrum
