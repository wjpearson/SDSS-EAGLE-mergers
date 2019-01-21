Identifying Galaxy Mergers in Observations and Simulations with Deep Learning
=============================================================================
This repo contains the code used to create, train, validate and test the 
convolutional neural networks presented in Pearson et al. 2019. The abstract 
of the paper can be found below.

The code runs on Python 3.6.1. Earlier (or later) versions of Python may work 
but have not been tested.

The requirements.txt include the python packages and version used to run these 
code. These are not the most up to date versions (at the time of writing, 
tensorflow is now 1.12.0 where as 1.10.1 was used).

Code Usage
----------
**get_real_SDSS_noise.py** should be run first. This code creates the noise 
cutouts that are used when processing EAGLE images to look like SDSS images. 
It requires three bands of the same SDSS field plus the catalogue that 
contains the objects in that field and the tsField file for that field. 
**img_path**, **img_file_#**, **cat_path**, **cat_file**, **tsF_file** and 
**out_path** variables will need to be updated to match your system setup, file
 structure and files used.

**SDSS_EAGLE_stamps-hdf5.py** should be run after **get_real_SDSS_noise.py**. 
This code takes the raw EAGLE .hdf5 files and processes them to look like real 
SDSS observations. It requires .hdf5 files of EAGLE objects, SDSS PSF files,
SDSS noise (generated with **get_real_SDSS_noise.py**) and a catalogue with 
redshifts of observed galaxies, if you wish to match the observed redshift 
distribution. **path**, **division**, **path_psf**, **redshift_file**, 
**z_col** and **noise_path** variables will need to be updated to match your 
system setup, file structure and files used. The paths in **psf_g**, **psf_r**,
 and **psf_i** will also need to be updated. The **ch** and **projection** 
variables will also need to be updated to match the channels and projections 
you wish to use.

**Train_network.py** is the code used to create and train the CNN. This can 
be run once the SDSS and EAGLE cutouts have been made. **path**, **division**, 
**extn**, **outfile** and **edge_cut** vairiables will need to be changed to 
match your file structure and data.

**Test_network.py** and **Test_network-cross_application.py** are run last. 
These codes calculate the statistics for the networks at validation and test 
time. **Test_network-cross_application.py** is also set up to run all the 
objects through the networks. As with **Train_network.py**, **path**, 
**division**, **extn**, **outfile** and **edge_cut** vairiables will need to 
be changed to match your file structure and data.

If you have any problems using these code, feel free to contact me.

Abstract
--------
_Context_. Mergers are an important aspect of galaxy growth and evolution. 
With large upcoming surveys, such as Euclid and LSST, fast, ecient and 
accurate techniques are needed to identify galaxy mergers for further study.

_Aims_. We aim to test wether deep learning techniques can be used to 
reproduce visual classification of observations, reproduce the differences that 
there may be between observation and simulation classifications.

_Methods_. A convolutional neural network was developed and trained twice, 
once with observations from SDSS and once with simulated galaxies from EAGLE, 
processed to mimic the SDSS observations. The accuracy of these networks were 
determined using an unused subset of the images. The SDSS images were also 
passed through the simulation trained network and the EAGLE images through the 
observation trained network.

_Results_. The observationally trained network achieves an accuracy of 91.5% 
while the simulation trained network achieves 74.4% on the SDSS and EAGLE 
images respectively. Passing the SDSS images through the simulation trained 
network was less successful, only achieving an accuracy of 64.3%, while 
passing the EAGLE images through the observation network was very poor, 
achieving an accuracy of only 49.7% with preferential assignment to the 
non-merger classification.

_Conclusions_. The networks trained and tested with the same data perform the 
best, with observations performing better than simulations, the latter a 
result of a more pure but less complete merger sample for the observations. 
Passing SDSS observations through the simulation trained network has proven to 
work, providing tantalising prospects for using simulation trained networks 
for galaxy identification in large surveys without needing an observational 
training set. However, an observationally trained network is still currently
the best deep learning method to interpret observational data.
