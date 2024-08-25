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

import parse_dat_file as parser

def main(targetFilepath, baraffeFilepath, mergedFilepath):    
    
    # Open files and extract data tables
    # baraffeData = fits.open(baraffeFilepath)[1].data
    baraffeData = parser.parse_dat_file(baraffeFilepath)
    mergedData = fits.open(mergedFilepath)[1].data

    # Create column array
    dataCols = [[], [], [], [], [], [], [], []]
    
    # Populate columns
    dataCols = populateDataColumns(dataCols, baraffeData, mergedData)

    # Define column datatype and convert to fits column
    fitsCols = [[], [], [], [], [], [], [], []]
    fitsCols = formatFitsCols(dataCols, fitsCols)

    # Package into fits table and write to file
    cols = fits.ColDefs([fitsCols[0], fitsCols[1], fitsCols[2],
                            fitsCols[3], fitsCols[4], fitsCols[5],
                            fitsCols[6], fitsCols[7]])
    
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(targetFilepath)

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

    for i in range(len(mergedData)):
        if (mergedData[i][0] >= 0.07):
            dataCols[0].append(mergedData[i][0])
            dataCols[1].append(mergedData[i][1])
            dataCols[2].append(mergedData[i][2])
            dataCols[3].append(mergedData[i][3])
            dataCols[4].append(mergedData[i][4])
            dataCols[5].append(mergedData[i][5])
            dataCols[6].append(mergedData[i][6])
            dataCols[7].append(mergedData[i][7])

    return dataCols

def formatFitsCols(dataCols, fitsCols):
    for i in range(6):
        colName = 'col' + str(i + 1)
        fitsCols[i] = fits.Column(name=colName, format='D', array=dataCols[i])

    fitsCols[6] = fits.Column(name="col7", format="K", array=dataCols[6])
    fitsCols[7] = fits.Column(name="col8", format="12A", array=dataCols[7])

    return fitsCols
