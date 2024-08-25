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
from astropy.table import Table
import numpy as np

def main(targetFilepath, baraffeFilepath, mergedFilepath):    
    
    # Open files and extract data tables
    baraffeData = fits.open(baraffeFilepath)[1].data
    mergedData = fits.open(mergedFilepath)[1].data

    # Create column array
    dataCols = [[], [], [], [], [], [], [], []]
    
    # Populate columns
    dataCols = populateDataColumns(dataCols, baraffeData, mergedData)

    # Define column datatype and convert to fits column
    fitsCols = [[], [], [], [], [], [], [], []]

    
'''
    Table formats: Mass, logT, logL, logG, logT_WR, M_curr, phase, source
        For our purposes, Mass and M_curr are the same, logT and logT_WR are the same, 
        phase is always 1, and source is always Baraffe
'''
def populateDataColumns(dataCols, baraffeData, mergedData):
    for i in range(len(baraffeData)):

    for i in range(len(mergedData)):
