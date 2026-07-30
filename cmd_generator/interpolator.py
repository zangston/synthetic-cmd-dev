import numpy as np


# Precondition:
# - isochrone grid and age array (in log years) are initialized
# - isochrones include magnitude columns for all requested filters
#
# Parameters:
# - age: desired stellar age in megayears
# - mass: desired stellar mass
# - iso_grid: sequence of SPISEA isochrones
# - age_arr: log10(age in years) for each isochrone
# - filters: sequence of magnitude-column names
#
# Returns:
# [
#     luminosity,
#     effective_temperature,
#     log_surface_gravity,
#     magnitude_filter_1,
#     magnitude_filter_2,
#     ...
# ]
#
# Returns None if the requested age or mass is outside the grid, or if
# interpolation cannot be performed.
def interpolate(age, mass, iso_grid, age_arr, filters):
    filters = list(filters)

    if len(filters) == 0:
        raise ValueError("At least one filter must be supplied.")

    if len(iso_grid) != len(age_arr):
        raise ValueError(
            "iso_grid and age_arr must contain the same number of entries."
        )

    # Identify the isochrones that bracket the requested age.
    idx_arr = findIsoIdx(age, age_arr)

    if idx_arr is None:
        return None

    a1, a2 = idx_arr

    # Interpolate in mass along the first isochrone.
    s1 = isoInterp(
        mass=mass,
        age_idx=a1,
        iso_grid=iso_grid,
        filters=filters,
    )

    if s1 is None:
        return None

    # Exact age match: no interpolation in age is necessary.
    #
    # s1 has the form:
    # [
    #     mass,
    #     luminosity,
    #     teff,
    #     logg,
    #     magnitude_1,
    #     ...
    # ]
    #
    # The returned result excludes mass.
    if a1 == a2:
        return np.round(
            np.asarray(s1[1:], dtype=float),
            decimals=3,
        )

    # Interpolate in mass along the second isochrone.
    s2 = isoInterp(
        mass=mass,
        age_idx=a2,
        iso_grid=iso_grid,
        filters=filters,
    )

    if s2 is None:
        return None

    # Interpolate the equal-mass reference stars in age.
    return ageInterp(
        age=age,
        s1=s1,
        a1=a1,
        s2=s2,
        a2=a2,
        age_arr=age_arr,
    )


# Precondition:
# - age_arr contains log10(age in years), sorted in increasing age
#
# Parameters:
# - age: desired age in megayears
# - age_arr: log10(age in years) for each isochrone
#
# Returns:
# - [idx, idx] for an exact age match
# - [idx_lo, idx_hi] for two isochrones bracketing the requested age
# - None if the age lies outside the available range
def findIsoIdx(age, age_arr):
    age_arr = np.asarray(age_arr, dtype=float)

    if age_arr.ndim != 1 or len(age_arr) == 0:
        raise ValueError("age_arr must be a non-empty one-dimensional array.")

    if not np.isfinite(age):
        return None

    ages_myr = np.power(10.0, age_arr) / 1.0e6

    if np.any(~np.isfinite(ages_myr)):
        raise ValueError("age_arr produced non-finite ages.")

    if np.any(np.diff(ages_myr) < 0):
        raise ValueError("age_arr must be sorted in increasing age.")

    # Outside available age range.
    if age < ages_myr[0] or age > ages_myr[-1]:
        return None

    # Exact match to an isochrone age.
    exact = np.where(
        np.isclose(
            ages_myr,
            age,
            rtol=0.0,
            atol=1.0e-10,
        )
    )[0]

    if len(exact) > 0:
        idx = int(exact[0])
        return [idx, idx]

    # Find adjacent ages bracketing the requested age.
    idx_hi = int(np.searchsorted(ages_myr, age))
    idx_lo = idx_hi - 1

    if idx_lo < 0 or idx_hi >= len(ages_myr):
        return None

    return [idx_lo, idx_hi]


# Precondition:
# - iso_grid[age_idx] is a valid SPISEA isochrone
# - all requested filters are present in iso_grid[age_idx].points
#
# Parameters:
# - mass: desired stellar mass
# - age_idx: index of the isochrone to interpolate along
# - iso_grid: sequence of SPISEA isochrones
# - filters: sequence of magnitude-column names
#
# Returns:
# [
#     mass,
#     luminosity,
#     effective_temperature,
#     log_surface_gravity,
#     magnitude_filter_1,
#     magnitude_filter_2,
#     ...
# ]
#
# Returns None if the requested mass is outside the isochrone's mass range
# or if a neighboring interpolation point cannot be found.
def isoInterp(mass, age_idx, iso_grid, filters):
    if not np.isfinite(mass):
        return None

    if age_idx < 0 or age_idx >= len(iso_grid):
        raise IndexError(
            f"age_idx={age_idx} is outside iso_grid with "
            f"{len(iso_grid)} entries."
        )

    iso = iso_grid[age_idx]

    if iso is None:
        return None

    points = iso.points
    required_columns = [
        "mass",
        "L",
        "Teff",
        "logg",
        *filters,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in points.colnames
    ]

    if missing_columns:
        raise KeyError(
            f"Isochrone at age index {age_idx} is missing columns "
            f"{missing_columns}. Available columns are {points.colnames}."
        )

    iso_mass_values = np.asarray(
        points["mass"],
        dtype=float,
    )

    finite_mass_mask = np.isfinite(iso_mass_values)

    if not np.any(finite_mass_mask):
        return None

    valid_indices = np.where(finite_mass_mask)[0]
    valid_masses = iso_mass_values[finite_mass_mask]

    mass_min = np.min(valid_masses)
    mass_max = np.max(valid_masses)

    # Requested mass lies outside the isochrone.
    if mass < mass_min or mass > mass_max:
        return None

    # Find the closest tabulated mass.
    local_nearest_idx = int(
        np.argmin(np.abs(valid_masses - mass))
    )
    s1_idx = int(valid_indices[local_nearest_idx])

    s1_mass = float(iso_mass_values[s1_idx])

    # Select the adjacent mass point on the opposite side of the target.
    if s1_mass < mass:
        candidate_indices = valid_indices[valid_indices > s1_idx]

        if len(candidate_indices) == 0:
            return None

        s2_idx = int(candidate_indices[0])

    elif s1_mass > mass:
        candidate_indices = valid_indices[valid_indices < s1_idx]

        if len(candidate_indices) == 0:
            return None

        s2_idx = int(candidate_indices[-1])

    else:
        # Exact mass match. Return the tabulated values directly.
        return _extract_isochrone_star(
            points=points,
            row_idx=s1_idx,
            filters=filters,
            requested_mass=mass,
        )

    s1 = _extract_isochrone_star(
        points=points,
        row_idx=s1_idx,
        filters=filters,
    )

    s2 = _extract_isochrone_star(
        points=points,
        row_idx=s2_idx,
        filters=filters,
    )

    if s1 is None or s2 is None:
        return None

    s1_mass = s1[0]
    s2_mass = s2[0]

    denom = s2_mass - s1_mass

    if np.isclose(denom, 0.0):
        result = s1.copy()
        result[0] = float(mass)
        return result

    w1 = (s2_mass - mass) / denom
    w2 = 1.0 - w1

    # Interpolate every property after mass:
    #
    # luminosity, Teff, logg, and every requested magnitude.
    interpolated_properties = [
        w1 * value1 + w2 * value2
        for value1, value2 in zip(s1[1:], s2[1:])
    ]

    result = [
        float(mass),
        *interpolated_properties,
    ]

    return result


# Parameters:
# - age: desired age in megayears
# - s1, s2: equal-mass stars interpolated along adjacent isochrones
# - a1, a2: indices of the corresponding isochrones
# - age_arr: log10(age in years) for each isochrone
#
# s1 and s2 have the form:
# [
#     mass,
#     luminosity,
#     Teff,
#     logg,
#     magnitude_1,
#     magnitude_2,
#     ...
# ]
#
# Returns:
# [
#     luminosity,
#     Teff,
#     logg,
#     magnitude_1,
#     magnitude_2,
#     ...
# ]
def ageInterp(age, s1, a1, s2, a2, age_arr):
    age_arr = np.asarray(age_arr, dtype=float)

    age1 = np.power(10.0, age_arr[a1]) / 1.0e6
    age2 = np.power(10.0, age_arr[a2]) / 1.0e6

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    if len(s1) != len(s2):
        raise ValueError(
            "s1 and s2 must contain the same number of properties."
        )

    if len(s1) < 5:
        raise ValueError(
            "Interpolated stars must contain mass, luminosity, Teff, "
            "logg, and at least one magnitude."
        )

    denom = age2 - age1

    if np.isclose(denom, 0.0):
        return np.round(
            s1[1:],
            decimals=3,
        )

    w1 = (age2 - age) / denom
    w2 = 1.0 - w1

    # Exclude mass from the returned values. Mass is already fixed and
    # identical for the two age-interpolated reference stars.
    result = (
        w1 * s1[1:]
        + w2 * s2[1:]
    )

    return np.round(
        result,
        decimals=3,
    )


def _extract_isochrone_star(
    points,
    row_idx,
    filters,
    requested_mass=None,
):
    """Extract one tabulated isochrone row in the internal star format.

    Returns:
    [
        mass,
        luminosity,
        Teff,
        logg,
        magnitude_1,
        magnitude_2,
        ...
    ]

    Returns None if any required value is non-finite.
    """
    mass = float(points[row_idx]["mass"])

    if requested_mass is not None:
        mass = float(requested_mass)

    luminosity = float(points[row_idx]["L"])
    teff = float(points[row_idx]["Teff"])
    logg = float(points[row_idx]["logg"])

    magnitudes = [
        float(points[row_idx][filt])
        for filt in filters
    ]

    values = [
        mass,
        luminosity,
        teff,
        logg,
        *magnitudes,
    ]

    if not np.all(np.isfinite(values)):
        return None

    return values