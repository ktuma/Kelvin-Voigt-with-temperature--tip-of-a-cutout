# Kelvin-Voigt with temperature -- tip of a cutout
Finite element implementation in FEniCS. This code was used for all simulations in our paper "Temperature field and heat generation at the tip of a cutout in a viscoelastic solid body undergoing loading" (https://arxiv.org/abs/2012.06204).

The code is based on FEniCS 2019.1.0 (https://fenicsproject.org/) using Python3 environment. It consists of two files:
* kelvin-voigt_temperature.py: script itself
* input_parameters: input parameters for the script (timestep, output directory, material parameters)

To run the simulation use <code>python3 kelvin-voigt_temperature.py input_parameters</code>.

In the output directory it generates several files:
* v.xdmf, v.h5; u.xdmf, u.h5; th.xdmf, th.h5: HDF5 data readable by Paraview (https://www.paraview.org/)
* temperature.csv: CSV file containing dependence of temperature at three different locations on time
* fields_time1.0.csv, fields_time2.0.csv, fields_time2.5.csv, fields_time10.0.csv: CSV file containing the temperature and displacement field in the zoomed area (defined in input parameters file) at times 1.0 s, 2.0 s, 2.5 s and 10.0 s.

For support please contact Karel Tůma (ktuma@karlin.mff.cuni.cz).
