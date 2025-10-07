import os
import csv
import logging
import argparse
from chi_squared_fitter import ChiSquaredFitter

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Run chi-squared fitting for JWST or JWST+HST photometry.")
parser.add_argument("--dat", type=str, required=True, help="Path to joined photometry file (with true AV, mass).")
parser.add_argument("--mode", type=str, choices=["jwst", "jwst_hst"], required=True,
                    help="Choose fitting mode: jwst or jwst_hst")
args = parser.parse_args()

# --- Configure logging ---
log_file = f"fit_progress_{args.mode}.log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# --- Configure fitter ---
if args.mode == "jwst":
    output_csv = "fit_results_jwst.csv"
    fitter = ChiSquaredFitter(
        filt_list=['jwst,F162M','jwst,F182M','jwst,F200W'],
        filters=['m_jwst_F162M','m_jwst_F182M','m_jwst_F200W'],
        output_dir='./fit_plots_jwst'
    )
elif args.mode == "jwst_hst":
    output_csv = "fit_results_jwst_hst.csv"
    fitter = ChiSquaredFitter(
        filt_list=['jwst,F162M','jwst,F182M','jwst,F200W','wfc3,ir,f125w'],
        filters=['m_jwst_F162M','m_jwst_F182M','m_jwst_F200W','m_hst_f125w'],
        output_dir='./fit_plots_jwst_hst'
    )

# --- Load data ---
with open(args.dat, "r") as f:
    all_lines = [ln for ln in f.readlines() if not ln.startswith("#")]

# --- Parse line helper ---
def parse_line(line, mode):
    parts = [float(x) for x in line.strip().split()]
    f162m, err_f162m = parts[2], parts[3]
    f182m, err_f182m = parts[4], parts[5]
    f200w, err_f200w = parts[6], parts[7]
    f125w, err_f125w = parts[8], parts[9]
    av, mass = parts[14], parts[15]

    if mode == "jwst":
        mags = [f162m, f182m, f200w]
        errs = [err_f162m, err_f182m, err_f200w]
    else:  # jwst_hst
        mags = [f162m, f182m, f200w, f125w]
        errs = [err_f162m, err_f182m, err_f200w, err_f125w]

    return mags, errs, av, mass

# --- Resume from existing results ---
processed_indices = set()
if os.path.exists(output_csv):
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_indices.add(int(row["index"]))
    logging.info(f"Loaded {len(processed_indices)} processed indices from {output_csv}")
else:
    logging.info("No existing results file found, starting fresh.")

# --- Run fitting ---
write_header = not os.path.exists(output_csv)
with open(output_csv, "a", newline="") as csvfile:
    fieldnames = ['index','best_mass','best_AV','mass_min','mass_max','AV_min','AV_max','intersects','min_chi2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_lines = len(all_lines)
    for idx, line in enumerate(all_lines):
        if idx in processed_indices:
            logging.info(f"Skipping line {idx}, already processed")
            continue

        mags, errs, av_true, mass_true = parse_line(line, args.mode)

        # Skip bad rows
        if any(m > 90 for m in mags) or av_true <= 0 or mass_true <= 0:
            logging.info(f"Skipping line {idx}, invalid values (mag>90 or AV/mass<=0)")
            continue

        try:
            logging.info(f"Processing line {idx+1}/{total_lines} ...")
            result = fitter.analyze_line(idx, mags, errs, av_true, mass_true)

            writer.writerow(result)
            csvfile.flush()

        except Exception as e:
            logging.error(f"Error processing line {idx}: {e}", exc_info=True)

logging.info("Processing complete.")