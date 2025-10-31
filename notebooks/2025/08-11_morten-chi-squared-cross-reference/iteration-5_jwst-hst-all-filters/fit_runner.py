import os
import csv
import logging
import argparse
from chi_squared_fitter import ChiSquaredFitter

parser = argparse.ArgumentParser(description="Run chi-squared fitting for JWST or JWST+HST photometry.")
parser.add_argument("--dat", type=str, required=True)
parser.add_argument("--mode", type=str, choices=["jwst", "jwst_hst"], required=True)
args = parser.parse_args()

log_file = f"fit_progress_{args.mode}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

if args.mode == "jwst":
    output_csv = "fit_results_jwst.csv"
    fitter = ChiSquaredFitter(
        filt_list=['jwst,F162M','jwst,F182M','jwst,F200W'],
        filters=['m_jwst_F162M','m_jwst_F182M','m_jwst_F200W'],
        output_dir='./fit_plots_jwst'
    )
else:  # All JWST + all HST filters
    output_csv = "fit_results_jwst_hstall.csv"
    fitter = ChiSquaredFitter(
        filt_list=['jwst,F162M','jwst,F182M','jwst,F200W',
                   'wfc3,ir,f125w','wfc3,ir,f139m','wfc3,ir,f160w'],
        filters=['m_jwst_F162M','m_jwst_F182M','m_jwst_F200W',
                 'm_hst_f125w','m_hst_f139m','m_hst_f160w'],
        output_dir='./fit_plots_jwst_hstall'
    )

with open(args.dat, "r") as f:
    all_lines = [ln for ln in f.readlines() if not ln.startswith("#")]

def parse_line(line, mode):
    parts = [float(x) for x in line.strip().split()]
    f162m, e162 = parts[2], parts[3]
    f182m, e182 = parts[4], parts[5]
    f200w, e200 = parts[6], parts[7]
    f125w, e125 = parts[8], parts[9]
    f139m, e139 = parts[10], parts[11]
    f160w, e160 = parts[12], parts[13]
    av, mass = parts[14], parts[15]

    if mode == "jwst":
        mags = [f162m, f182m, f200w]
        errs = [e162, e182, e200]
    else:
        mags = [f162m, f182m, f200w, f125w, f139m, f160w]
        errs = [e162, e182, e200, e125, e139, e160]
    return mags, errs, av, mass

processed_indices = set()
if os.path.exists(output_csv):
    with open(output_csv) as f:
        processed_indices = {int(r["index"]) for r in csv.DictReader(f)}

write_header = not os.path.exists(output_csv)
with open(output_csv, "a", newline="") as csvfile:
    fieldnames = ['index','best_mass','best_AV','mass_min','mass_max','AV_min','AV_max','intersects','min_chi2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header: writer.writeheader()

    for idx, line in enumerate(all_lines):
        if idx in processed_indices: continue
        mags, errs, av_true, mass_true = parse_line(line, args.mode)
        if any((m > 90 or e > 90 or e <= 0) for m, e in zip(mags, errs)) or av_true <= 0 or mass_true <= 0:
            continue
        try:
            result = fitter.analyze_line(idx, mags, errs, av_true, mass_true)
            writer.writerow(result)
            csvfile.flush()
        except Exception as e:
            logging.error(f"Error processing line {idx}: {e}", exc_info=True)