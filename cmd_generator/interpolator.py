import numpy as np

# Precondition: isochrone grid and age array (in log years) are initialized, isochrones include magnitudes for specified filters
# Params: desired age (in megayears), desired mass, isochrone grid, age array, two filters to use for CMD
# Return: interpolated star with luminosity in watts, effective temp in kelvin, log surface gravity, and magnitudes for two filters
def interpolate(age, mass, iso_grid, age_arr, filters):
    # Identify which isochrones to interpolate along
    idx_arr = findIsoIdx(age, age_arr)
    if idx_arr is None:
        return None

    a1, a2 = idx_arr

    # Perform interpolation along isochrones to find reference stars of equal mass
    s1 = isoInterp(mass, a1, iso_grid, filters)
    if s1 is None:
        return None

    # Exact age match: no need to interpolate in age
    if a1 == a2:
        s = [s1[1], s1[2], s1[3], s1[4], s1[5]]
        for i in range(len(s)):
            s[i] = np.round(s[i], decimals=3)
        return s

    s2 = isoInterp(mass, a2, iso_grid, filters)
    if s2 is None:
        return None

    # Interpolate along age between s1 and s2
    return ageInterp(age, s1, a1, s2, a2, age_arr)


# Precondition: age array initialized
# Params: desired age of interpolated star, array of ages associated with isochrone grid
# Return: array containing indices of two adjacent isochrones that bracket the desired age
def findIsoIdx(age, age_arr):
    ages_myr = np.power(10, age_arr) / 1e6

    # Outside available age range
    if age < ages_myr[0] or age > ages_myr[-1]:
        return None

    # Exact match to an isochrone age
    exact = np.where(np.isclose(ages_myr, age, rtol=0.0, atol=1e-10))[0]
    if len(exact) > 0:
        idx = exact[0]
        return [idx, idx]

    # Bracket using searchsorted
    idx_hi = np.searchsorted(ages_myr, age)
    idx_lo = idx_hi - 1

    if idx_lo < 0 or idx_hi >= len(ages_myr):
        return None

    return [idx_lo, idx_hi]


# Precondition: isochrone grid initialized, filters specified
# Params: desired mass of interpolated star, grid of isochrones to interpolate along, filter magnitudes to interpolate between
# Return: array of properties for a star with a certain mass and age
def isoInterp(mass, age_idx, iso_grid, filters):
    iso_mass_values = iso_grid[age_idx].points['mass']

    # Check if the desired mass is within the range of the isochrone
    if mass < np.min(iso_mass_values) or mass > np.max(iso_mass_values):
        return None

    # extract closest star to mass
    s1_idx = np.where(np.abs(iso_grid[age_idx].points['mass'] - mass) ==
                      np.min(np.abs(iso_grid[age_idx].points['mass'] - mass)))[0].item()

    s1_mass = np.round(iso_grid[age_idx].points[s1_idx]['mass'], decimals=3)
    s1_lum = np.round(iso_grid[age_idx].points[s1_idx]['L'], decimals=3)
    s1_teff = np.round(iso_grid[age_idx].points[s1_idx]['Teff'], decimals=3)
    s1_logg = np.round(iso_grid[age_idx].points[s1_idx]['logg'], decimals=3)
    s1_filt1 = np.round(iso_grid[age_idx].points[s1_idx][filters[0]], decimals=3)
    s1_filt2 = np.round(iso_grid[age_idx].points[s1_idx][filters[1]], decimals=3)
    s1 = [s1_mass, s1_lum, s1_teff, s1_logg, s1_filt1, s1_filt2]

    # extract next star to interpolate with
    if s1_mass < mass:
        s2_idx = s1_idx + 1
    else:
        s2_idx = s1_idx - 1

    # Guard against edge-of-array cases
    if s2_idx < 0 or s2_idx >= len(iso_grid[age_idx].points):
        return None

    s2_mass = np.round(iso_grid[age_idx].points[s2_idx]['mass'], decimals=3)
    s2_lum = np.round(iso_grid[age_idx].points[s2_idx]['L'], decimals=3)
    s2_teff = np.round(iso_grid[age_idx].points[s2_idx]['Teff'], decimals=3)
    s2_logg = np.round(iso_grid[age_idx].points[s2_idx]['logg'], decimals=3)
    s2_filt1 = np.round(iso_grid[age_idx].points[s2_idx][filters[0]], decimals=3)
    s2_filt2 = np.round(iso_grid[age_idx].points[s2_idx][filters[1]], decimals=3)
    s2 = [s2_mass, s2_lum, s2_teff, s2_logg, s2_filt1, s2_filt2]

    denom = (s2_mass - s1_mass)
    if np.isclose(denom, 0.0):
        return [mass, s1_lum, s1_teff, s1_logg, s1_filt1, s1_filt2]

    w1 = (s2_mass - mass) / denom
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


# Precondition: age array is initialized, indices of isochrones being interpolated between are identified
# Params: desired age of interpolated star, arrays representing stars to interpolate between of equal mass,
#         array of ages associated with isochrone grid, ages to interpolate between
# Return: array of properties for star of a given age
def ageInterp(age, s1, a1, s2, a2, age_arr):
    age1 = np.power(10, age_arr[a1]) / 1e6
    age2 = np.power(10, age_arr[a2]) / 1e6

    denom = age2 - age1
    if np.isclose(denom, 0.0):
        s = [s1[1], s1[2], s1[3], s1[4], s1[5]]
        for i in range(len(s)):
            s[i] = np.round(s[i], decimals=3)
        return s

    # using two stars of same mass and differing age, perform linear interpolation along age
    w1 = (age2 - age) / denom
    w2 = 1.0 - w1

    # Interpolate the properties
    s_lum = w1 * s1[1] + w2 * s2[1]
    s_teff = w1 * s1[2] + w2 * s2[2]
    s_logg = w1 * s1[3] + w2 * s2[3]
    s_filt1 = w1 * s1[4] + w2 * s2[4]
    s_filt2 = w1 * s1[5] + w2 * s2[5]

    # Store the interpolated values in s
    s = [s_lum, s_teff, s_logg, s_filt1, s_filt2]

    # Truncate values
    for i in range(len(s)):
        s[i] = np.round(s[i], decimals=3)

    return s