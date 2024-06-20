# import packages
import numpy as np
import pylab as py
import pdb
import matplotlib.pyplot as plt
import math

# import spisea packages
from spisea import synthetic, evolution, atmospheres, reddening, ifmr
from spisea.imf import imf, multiplicity

# import nbody6tools reader to read data
from nbody6tools import Reader

# import nbody62spisea to retrieve star ages
from nbody62spisea import converter

# colormap stuff for gradient plotting
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# import interpolator for calculating photometric values
import interpolator as interpolator

def generate_cmd(AKs, dist, evo_model, atm_func, red_law, filt_list, filters, iso_dir, metallicity, nbody_path):

    snapshot = Reader.read_snapshot(nbody_path, time = 1.5)

    # convert snapshot values from nbody units to astrophysical, extract masses
    snapshot.to_physical()
    cluster_table = converter.to_spicea_table(snapshot)

    masses = cluster_table['mass']
    level_ages = cluster_table['age']

    for i in range(len(level_ages)):
        level_ages[i] = level_ages[i] * 1e6

    # convert log ages from log(Myr) to log(yr)
    log_ages = np.log10(level_ages)

    # find index of first binary, create array of companion masses
    first_binary = 0
    for i in range(len(cluster_table)):
        if cluster_table['isMultiple'][i] == 1.0:
            first_binary = i
            break

    compMasses = []
    for i in range(len(cluster_table)):
        if cluster_table['isMultiple'][i] == 1.0:
            compMasses.append(cluster_table['compMass'][i][0])

    increment = 0.5e6
    start = (min(level_ages) // increment) * increment
    end = (max(level_ages) // increment) * increment + 2 * increment

    # Create age arrays
    level_age_arr = np.arange(start, end, increment)
    log_age_arr = np.log10(level_age_arr)

    # create isochrone grid - if this is the first time, then this is going to take an hour lmfao
    instances = np.empty(len(log_age_arr), dtype=object)

    for i in range(len(log_age_arr)):
        my_iso = synthetic.IsochronePhot(log_age_arr[i], AKs, dist, metallicity=metallicity,
                                evo_model=evo_model, atm_func=atm_func,
                                red_law=red_law, filters=filt_list,
                                    iso_dir=iso_dir)
        instances[i] = my_iso
            
    print("isochrone generation done")

    # identify array indeces, create array for labels and colors
    idx_arr = range(0, len(log_age_arr))
    if len(idx_arr) > 3:
        idx_arr = idx_arr[::2]

    # Plot CMD
    fig, axes = py.subplots(figsize=(15, 10))
    py.subplot(1, 2, 1)

    # Define a colormap
    cmap = plt.get_cmap('coolwarm')

    for i in range(len(log_age_arr)):
        color = cmap(i / (len(log_age_arr) - 1))  # Assign color based on index and colormap
        py.plot(instances[i].points[filters[0]] - instances[i].points[filters[1]], 
            instances[i].points[filters[1]], color=color, label='')
        
    py.xlabel('F115W - F200W')
    py.ylabel('F200W')
    py.gca().invert_yaxis()

    # Create colorbar legend
    norm = Normalize(vmin=min(level_age_arr), vmax=max(level_age_arr))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Age (millions of years)')
    cbar.set_ticks(level_age_arr)
    cbar.set_ticklabels([f'{age/1e6:.1f}' for age in level_age_arr])

    level_ages_myr = level_ages / 1e6
    # print(level_ages_myr)

    # create array of stars
    stars = np.empty(len(cluster_table), dtype=object)

    # perform interpolation for each star
    for i in range(len(stars)):
        # print(str(i) + " " + str(level_ages_myr[i]) + " " + str(masses[i]))
        stars[i] = interpolator.interpolate(level_ages_myr[i], masses[i], instances, log_age_arr, filters)
        
    # convert luminosity values to solar luminosities
    watts_to_lsun = 1.0 / (3.846e26) # conversion factor for watts to Lsun

    for i in range(len(stars)):
        if stars[i] is None:
            continue
        stars[i][0] = stars[i][0] * watts_to_lsun

    # perform interpolation on companion stars
    companions = np.empty(len(compMasses), dtype=object)

    for i in range(len(companions)):
        if stars[i + first_binary] is None:
            continue
        companions[i] = interpolator.interpolate(level_ages_myr[i + first_binary], compMasses[i], instances, log_age_arr, filters)

    for i in range(len(companions)):
        if companions[i] is None:
            continue
        companions[i][0] = companions[i][0] * watts_to_lsun

    # plot primaries and companions separately
    fig, axes = py.subplots(figsize=(15, 10))

    # plot primaries
    py.subplot(1, 2, 1)

    # Define a colormap
    cmap = plt.get_cmap('coolwarm')

    for i in range(len(log_age_arr)):
        color = cmap(i / (len(log_age_arr) - 1))  # Assign color based on index and colormap
        py.plot(instances[i].points[filters[0]] - instances[i].points[filters[1]], 
            instances[i].points[filters[1]], color=color, label='')
        
    for i in range(0, first_binary):
        if stars[i] is None:
            continue
        py.plot(stars[i][3] - stars[i][4], stars[i][4], marker='o', markersize=1, color='k')
        
    py.xlabel('F115W - F200W')
    py.ylabel('F200W')
    py.gca().invert_yaxis()

    py.title('CMD of primary stars only')

    # Create colorbar legend
    norm = Normalize(vmin=min(level_age_arr), vmax=max(level_age_arr))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Age (millions of years)')
    cbar.set_ticks(level_age_arr)
    cbar.set_ticklabels([f'{age/1e6:.1f}' for age in level_age_arr])

    # plot companions
    py.subplot(1, 2, 2)

    # Define a colormap
    cmap = plt.get_cmap('coolwarm')

    for i in range(len(log_age_arr)):
        color = cmap(i / (len(log_age_arr) - 1))  # Assign color based on index and colormap
        py.plot(instances[i].points[filters[0]] - instances[i].points[filters[1]], 
            instances[i].points[filters[1]], color=color, label='')
        
    for i in range(len(companions)):
        if companions[i] is None:
            continue
        py.plot(companions[i][3] - companions[i][4], companions[i][4], marker='o', markersize=1, color='k')
        
    py.xlabel('F115W - F200W')
    py.ylabel('F200W')
    py.gca().invert_yaxis()

    py.title('CMD of companion stars only')

    # Create colorbar legend
    norm = Normalize(vmin=min(level_age_arr), vmax=max(level_age_arr))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Age (millions of years)')
    cbar.set_ticks(level_age_arr)
    cbar.set_ticklabels([f'{age/1e6:.1f}' for age in level_age_arr])
        
    plt.show()

    # combine magnitudes on binary stars
    unresolved_binaries = np.empty(len(companions), dtype=object)

    for i in range(len(companions)):
        if stars[i + first_binary] is None or companions[i] is None:
            continue
        
        flux11 = np.power(10, stars[i + first_binary][3] / -2.5) * 3.93e-10
        flux21 = np.power(10, companions[i][3] / -2.5) * 3.93e-10
        mag1 = -2.5 * np.log10((flux11 + flux21) / 3.93e-10)
        
        flux12 = np.power(10, stars[i + first_binary][4] / -2.5) * 5.74e-11
        flux22 = np.power(10, companions[i][4] / -2.5) * 5.74e-11
        mag2 = -2.5 * np.log10((flux12 + flux22) / 5.744e-11)
        
        unresolved_binaries[i] = [mag1, mag2]

    fig, axes = py.subplots(figsize=(15, 10))

    fig, axes = py.subplots(figsize=(15, 10))

    py.subplot(1, 2, 1)

    # CMD with unresolved binaries
    # Define a colormap
    cmap = plt.get_cmap('coolwarm')
    
    for i in range(len(log_age_arr)):
        color = cmap(i / (len(log_age_arr) - 1))  # Assign color based on index and colormap
        py.plot(instances[i].points[filters[0]] - instances[i].points[filters[1]], 
            instances[i].points[filters[1]], color=color, label='')

    # plot single stars first
    for i in range(0, first_binary):
        if stars[i] is None:
            continue
        py.plot(stars[i][3] - stars[i][4], stars[i][4], marker='o', markersize=1, color='k')

    # plot unresolve binaries
    for i in range(len(unresolved_binaries)):
        if unresolved_binaries[i] is None:
            continue
        py.plot(stars[first_binary + i][3] - stars[first_binary + i][4], 
                unresolved_binaries[i][1], marker='o', markersize=1, color='green')

    py.xlabel('F115W - F200W')
    py.ylabel('F200W')
    py.gca().invert_yaxis()
    py.title('CMD with unresolved binary systems')

    # Create colorbar legend
    norm = Normalize(vmin=min(level_age_arr), vmax=max(level_age_arr))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Age (millions of years)')
    cbar.set_ticks(level_age_arr)
    cbar.set_ticklabels([f'{age/1e6:.1f}' for age in level_age_arr])
        
    plt.show()
