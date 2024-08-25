'''
============= MERGE FILES WORKFLOW ===========
1. Params: target file name, old merged file name, baraffe file name
2. Populate columns
    0.01 - 0.07 Msun: Baraffe
    > 0.07 Msun: Merged model
3. Format columns, form into fits binary table
5. write to specified target file
6. return indication of success
'''
from astropy.io import fits
import numpy as np

def main(target_filepath, baraffe_filepath, merged_filepath):    
    baraffe_file = fits.open(baraffe_filepath)
    merged_file = fits.open(merged_filepath)

    baraffe_data = baraffe_file[1]
    print(baraffe_data)
    merged_data = merged_file[1]

# def populate_columns():

