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

    print(dataCols)

    # Define column datatype and convert to fits column
    fitsCols = [[], [], [], [], [], [], [], []]

    
'''
    Table formats: Mass, logT, logL, logG, logT_WR, M_curr, phase, source
        For our purposes, Mass and M_curr are the same, logT and logT_WR are the same, 
        phase is always 1, and source is always Baraffe
'''
def populateDataColumns(dataCols, baraffeData, mergedData):
    
    # Add every row from the Baraffe file for mass < 0.07 Msun
    for i in range(len(baraffeData)):
        if (baraffeData[i][0] < 0.07):
            dataCols[0].append(baraffeData[i][0])
            dataCols[1].append(baraffeData[i][1])
            dataCols[2].append(baraffeData[i][2])
            dataCols[3].append(baraffeData[i][3])
            dataCols[4].append(baraffeData[i][4])
            dataCols[5].append(baraffeData[i][5])
            dataCols[6].append(baraffeData[i][6])
            dataCols[7].append(baraffeData[i][7])

    # for i in range(len(mergedData)):

    return dataCols
