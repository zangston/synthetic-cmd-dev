import numpy as np
import csv

# Input files
jwst_file = "/scratch/wyz5rge/synthetic-hr/notebooks/2025/08-11_morten-chi-squared-cross-reference/winstonlist.dat"
joined_file = "/scratch/wyz5rge/synthetic-hr/notebooks/2025/08-11_morten-chi-squared-cross-reference/phot_joined.dat"

# Output file
output_file = "phot_joined_with_truth.dat"

# === Load JWST-only data ===
jwst_data = np.loadtxt(jwst_file)
# Format: x, y, F162, err, F182, err, F200, err, AV, mass
jwst_coords = jwst_data[:, :2]
jwst_truth = jwst_data[:, 8:10]  # AV, mass

# === Load joined data ===
joined_data = np.loadtxt(joined_file)
# Format: x, y, JWST mags/errs, HST mags/errs

# Mask: exclude rows where ANY HST magnitude > 90
# JWST = cols 2–7, HST mags = cols 8, 10, 12
hst_mags = joined_data[:, [8, 10, 12]]
mask = np.all(hst_mags < 90, axis=1)
joined_valid = joined_data[mask]

print(f"Original joined rows: {len(joined_data)}")
print(f"Valid joined rows (after masking): {len(joined_valid)}")

# === Match with JWST-only file ===
# We'll use a tolerance for (x,y) matching since floats may not be exact
tol = 1e-3
matched_rows = []

for row in joined_valid:
    x, y = row[0], row[1]
    # Find index in JWST file with matching coordinates
    diffs = np.abs(jwst_coords - np.array([x, y]))
    dist = np.sqrt((diffs**2).sum(axis=1))
    min_idx = np.argmin(dist)
    if dist[min_idx] < tol:
        av, mass = jwst_truth[min_idx]
        matched_rows.append(np.hstack([row, av, mass]))

matched_rows = np.array(matched_rows)

print(f"Matched rows: {len(matched_rows)}")

# === Save output ===
header = (
    "x y F162M err_F162M F182M err_F182M F200W err_F200W "
    "F125W err_F125W F139M err_F139M F160W err_F160W true_AV true_mass"
)

np.savetxt(output_file, matched_rows, fmt="%.6f", header=header)

print(f"Saved {len(matched_rows)} rows to {output_file}")