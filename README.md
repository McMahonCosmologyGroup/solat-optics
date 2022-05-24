# solat-optics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Near- and Far-field optical simulation of the Simons Observatory Large Aperture Telescope.  This is the repository associated with the publication: <i>The Simons Observatory: Characterizing the Large Aperture Telescope Receiver with Radio Holography</i>.  <br />

Author: Grace E. Chesmore

<p align="center">
     <img src="https://github.com/McMahonCosmologyGroup/solat-optics/blob/main/figures/lat34.png?raw=true" alt="centered image" width="50%"/>
</p>

## Installation Instructions:

1. `git clone git@github.com:McMahonCosmologyGroup/solat-optics.git`
2. `cd solat_optics/`
3. `python3 setup.py install --user`

## Tutorials
Three tutorials are available as Jupyter notebooks:
1. [Near-field](https://github.com/McMahonCosmologyGroup/solat-optics/tree/main/tutorials/latrt_holo_sim.ipynb)
2. [Far-field](https://github.com/McMahonCosmologyGroup/solat-optics/tree/main/tutorials/latrt_farfield_sim.ipynb)
3. [Quantifying Optical Systematics](https://github.com/McMahonCosmologyGroup/solat-optics/tree/main/tutorials/quant_systematics.ipynb)

### Near-field
Part 1 of the tutorial builds the surfaces of the optics tube which will eventually be used in the near-field simulation of the optics tube.  First ray-trace from the focal plane out through the window.  This toy-model is useful for predicting the beam path and size based on the receiver's position of the focal plane, however this will provide no diffraction information.  The user can drag / zoom / and hover over the ray-trace to see the labelled optics tube surfaces.

<p align="center">
     <img src="https://github.com/McMahonCosmologyGroup/solat-optics/blob/main/figures/raytrace1.png?raw=true" alt="centered image" width="50%"/>
</p>

Part 2 ray-traces from both the source and the receiver positions to the Lyot stop, located near the center of the optics tube, yielding the source and receiver electric fields $E_{\text{source}}$ and $E_{\text{rec}}$.  The Lyot stop defines the size of the beam and therefor we integrate the receiver and source electric fields over the area of the Lyot stop (where the pink and blue rays meet in the middle of the optics tube).

<p align="center">
     <img src="https://github.com/McMahonCosmologyGroup/solat-optics/blob/main/figures/raytrace2.png?raw=true" alt="centered image" width="50%"/>
</p>

Part 3 demonstrates how to run the simulation from the command line, and how to do so in parallel to speed up the simulation.  Doing so will save the simulation to a .txt file.  Lastly, the simulated near-field power and phase are read out from the text file and plotted.

### Far-field

### Quantifying Optical Systematics


## Contributions
If you have write access to this repository, please:
* create a new branch
* push your changes to that branch
* merge or rebase to get in sync with main
* submit a pull request on github
* If you do not have write access, create a fork of this repository and proceed as described above. For more details, see Contributing.

