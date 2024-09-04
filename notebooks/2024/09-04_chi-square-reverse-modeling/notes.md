Goal: given set of photometric values for a star, can we guess its age and mass?
Method: perform a chi-squared analysis against existing points in isochrone grid to find 
    candidate age-mass value pairs that minimize the chi-squared measure

# Algorithm:
1. Define isochrone grid using parameters matching Juan's simulation data
2. Create 2D array along age and mass, with each element containing an n-element array of magnitudes for n filters
3. Take in a particular star and its photometric values
    - (for this purpose, pick a random star from the simulation and calculate its photometery using interpolation)
4. Perform a naive O(n^2) comparison, and print the mass-age value pairs for which the star's chi-squared is smallest
    - Degrees of freedom for chi-squared = n - 1
    - Can try to find a more efficient minimization algorithm later