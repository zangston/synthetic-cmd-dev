import numpy as np

# Precondition: isochrone grid and age array (in log years) are initialized, isochrones include magnitudes for specified filters
# Params: desired age (in megayears), desired mass, isochrone grid, age array, two filters to use for CMD
# Return: interpolated star with luminosity in watts, effective temp in kelvin, log surface gravity, and magnitudes for two filters
def interpolate(age, mass, iso_grid, age_arr, filters):

    # Identify which isochrones to interpolate along
    idx_arr = findIsoIdx(age, age_arr)
    a1 = idx_arr[0]
    a2 = idx_arr[1]

    # Perform interpolation along isochrones to find reference stars of equal mass
    s1 = isoInterp(mass, a1, iso_grid, filters)
    s2 = isoInterp(mass, a2, iso_grid, filters)

    # Interpolate along age between s1 and s2
    if s1 is not None and s2 is not None:
        s = ageInterp(age, s1, a1, s2, a2, age_arr)
        return s
    return None

# Precondition: age array initialized
# Params: desired age of interpolated star, array of ages associated with isochrone grid
# Return: array containing indeces of two adjacent isochrones that bracket the desired age
def findIsoIdx(age, age_arr):
    ret_idx = [0, 0]
    for i in range(len(age_arr) - 1):
        if np.power(10, age_arr[i]) <= age * 1000000 <= np.power(10, age_arr[i + 1]):
            ret_idx[0] = i
            ret_idx[1] = i + 1
            break
    return ret_idx

# Precondition: isochrone grid initialized, filters specified
# Params: desired mass of interpolated star, grid of isochrones to interpolate along, filter magnitudes to interpolate between
# Return: array of properties for a star with a certain mass and age
def isoInterp(mass, age_idx, iso_grid, filters):

    iso_mass_values = iso_grid[age_idx].points['mass']
    
    # Check if the desired mass is within the range of the isochrone
    if mass < min(iso_mass_values) or mass > max(iso_mass_values):
        # Handle the case where the desired mass is outside the range
        print(f"Desired mass {mass} is outside the range of the isochrone.")
        # You can return a default value or raise an exception based on your requirements
        return None

    # extract closest star to mass
    s1_idx = np.where(abs(iso_grid[age_idx].points['mass'] - mass) == min(abs(iso_grid[age_idx].points['mass'] - mass)))[0].item()
    s1_mass = np.round(iso_grid[age_idx].points[s1_idx]['mass'], decimals=3)
    s1_lum = np.round(iso_grid[age_idx].points[s1_idx]['L'], decimals=3)
    s1_teff = np.round(iso_grid[age_idx].points[s1_idx]['Teff'], decimals=3)
    s1_logg = np.round(iso_grid[age_idx].points[s1_idx]['logg'], decimals=3)
    s1_filt1 = np.round(iso_grid[age_idx].points[s1_idx][filters[0]], decimals=3)
    s1_filt2 = np.round(iso_grid[age_idx].points[s1_idx][filters[1]], decimals=3)
    s1 = [s1_mass, s1_lum, s1_teff, s1_logg, s1_filt1, s1_filt2]

    # extract next star to interpolate with
    if(s1_mass < mass):
        s2_idx = s1_idx + 1
    else:
        s2_idx = s1_idx - 1

    s2_mass = np.round(iso_grid[age_idx].points[s2_idx]['mass'], decimals=3)
    s2_lum = np.round(iso_grid[age_idx].points[s2_idx]['L'], decimals=3)
    s2_teff = np.round(iso_grid[age_idx].points[s2_idx]['Teff'], decimals=3)
    s2_logg = np.round(iso_grid[age_idx].points[s2_idx]['logg'], decimals=3)
    s2_filt1 = np.round(iso_grid[age_idx].points[s2_idx][filters[0]], decimals=3)
    s2_filt2 = np.round(iso_grid[age_idx].points[s2_idx][filters[1]], decimals=3)
    s2 = [s2_mass, s2_lum, s2_teff, s2_logg, s2_filt1, s2_filt2]

    w1 = (s2_mass - mass) / (s2_mass - s1_mass)
    w2 = 1.0 - w1

    # Interpolate the properties
    s_mass = mass
    s_lum = w1 * s1_lum + w2 * s2_lum
    s_teff = w1 * s1_teff + w2 * s2_teff
    s_logg = w1 * s1_logg + w2 * s2_logg
    s_filt1 = w1 * s1_filt1 + w2 * s2_filt1
    s_filt2 = w1 * s1_filt2 + w2 * s2_filt2

    # Store the interpolated values in s
    s = [s_mass, s_lum, s_teff, s_logg, s_filt1, s_filt2]
    return s

# Precondition: age array is intialized, indeces of isochrones being interpolated between are indentified
# Params: desired age of interpolated star, arrays representing stars to interpolate between of equal mass, 
#           array of ages associated with isochrone grid, ages to interpolate between
# Return: array of properties for star of a given age
def ageInterp(age, s1, a1, s2, a2, age_arr):
    # using two stars of same mass and differing age, perform linear interpolation along age
    w1 = (np.power(10, age_arr[a2]) / 1000000 - age) / (np.power(10, age_arr[a2]) / 1000000 - np.power(10, age_arr[a1]) / 1000000)
    w2 = 1.0 - w1

    # Interpolate the properties
    s_age = age
    s_lum = w1 * s1[1] + w2 * s2[1]
    s_teff = w1 * s1[2] + w2 * s2[2]
    s_logg = w1 * s1[3] + w2 * s2[3]
    s_filt1 = w1 * s1[4] + w2 * s2[4]
    s_filt2 = w1 * s1[5] + w2 * s2[5]

    # Store the interpolated values in s
    s = [s_lum, s_teff, s_logg, s_filt1, s_filt2]

    # Truncate values
    for i in range(len(s)):
        s[i] = np.round(s[i], decimals = 3)

    return s
