# Chi-Square minimization iteration 2 notes
Created 30 Sept 2024
## Findings from last iteration
### Points are way off
- s284 magnitudes were way off from the isochrone grid that I generated
- discrepancy is due to isochrone grid having been generated with no extinction, and data points not having been de-reddened

### Even if the data points were "true", the estimations look off too?
- minimization appears to not be working when viewing sample vs forecast points in the color-magnitude space
- see 'iteration1/chi-square-reverse-model-test-run' - even if the de-reddened point is correct, we should be seeing the estimated points be at the 0.8-0.9 solar mass bend in the isochrones, which minimizes the distance
- this deviation is most likely because the minimization algorithm i made takes into account all filters supplied

## Things to do
1. Create a magnitude-magnitude plot of the isochrone grid
2. Create plot with mass tracks to see how stars vary with age
3. Limit minimization algorithm down to only 2 filters