# correlate_2D
Correlate time series on regular meshes

```

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
