import os, csv, logging, argparse
from chi_squared_fitter import ChiSquaredFitter

parser = argparse.ArgumentParser(description="Run chi-squared fitting for JWST+HST F160W photometry.")
parser.add_argument("--dat", type=str, required=True)
args = parser.parse_args()

output_csv = "fit_results_jwst_hst_f160.csv"
fitter = ChiSquaredFitter(
    filt_list=['jwst,F162M','jwst,F182M','jwst,F200W','wfc3,ir,f160w'],
    filters=['m_jwst_F162M','m_jwst_F182M','m_jwst_F200W','m_hst_f160w'],
    output_dir='./fit_plots_jwst_hst_f160'
)

with open(args.dat, "r") as f:
    all_lines = [ln for ln in f.readlines() if not ln.startswith("#")]

def parse_line(line):
    p = [float(x) for x in line.strip().split()]
    mags = [p[2], p[4], p[6], p[12]]   # F162, F182, F200, F160
    errs = [p[3], p[5], p[7], p[13]]
    av, mass = p[14], p[15]
    return mags, errs, av, mass

processed = set()
if os.path.exists(output_csv):
    with open(output_csv) as f:
        processed = {int(r["index"]) for r in csv.DictReader(f)}

write_header = not os.path.exists(output_csv)
with open(output_csv, "a", newline="") as csvfile:
    fields = ['index','best_mass','best_AV','mass_min','mass_max','AV_min','AV_max','intersects','min_chi2']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    if write_header: writer.writeheader()

    for idx, line in enumerate(all_lines):
        if idx in processed: continue
        mags, errs, av_true, mass_true = parse_line(line)
        if any(m > 90 for m in mags) or any(e <= 0 or e > 90 for e in errs) or av_true <= 0 or mass_true <= 0:
            continue
        try:
            result = fitter.analyze_line(idx, mags, errs, av_true, mass_true)
            writer.writerow(result)
            csvfile.flush()
        except Exception as e:
            logging.error(f"Error line {idx}: {e}", exc_info=True)