# correlate_2D
Correlate time series on regular meshes

## Python and Dependencies
The code needs the python packages *netcdf4* and *progressbar*

If you know what you are doing, just make sure the aforementioned dependencies are installed. Otherwise do yourself a favor and download the Anaconda (https://store.continuum.io/cshop/anaconda/) Python distribution. It is a free scientific Python distribution bundling almost all necessary modules with a convenient installer (does not require root access!). Once installed assert that pip and conda point to the Anaconda installation folder (you may need to open a new terminal after installing Anaconda).

    $ conda install -c netcdf4 progressbar pip

## Usage
```
$ python correlate_2d.py -h

usage: correlate_2d.py [-h] [-o OUTPUT_FILE_NAME] [-v VARIABLE]
                       [-m MAX_LENGTH] [-d DEPTH_LAYER] [-k]
                       input_file_name

Calculate spatial correlation length in a 2D time series

positional arguments:
  input_file_name       NetCDF input file name. Should follow BLABLA convention

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE_NAME, --output_file_name OUTPUT_FILE_NAME
                        Output file name. Default is input_file_name_corr.nc
  -v VARIABLE, --variable VARIABLE
                        Variable to calculate correlation on
  -m MAX_LENGTH, --max_length MAX_LENGTH
                        Maximum correlation length. 
                        Only elements with a distance smaller max_length are 
                        correlated. Smaller max_length speed up the calculation
  -d DEPTH_LAYER, --depth_layer DEPTH_LAYER
                        Depth layer to correlate. 
  -k, --keep_original   Keep original variable in output file?
  
```
