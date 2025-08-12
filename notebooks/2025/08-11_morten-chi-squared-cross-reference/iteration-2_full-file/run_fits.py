import os
import csv
import logging
from chi_squared_fitter import ChiSquaredFitter

# Paths
dat_path = '/scratch/wyz5rge/synthetic-hr/notebooks/2025/08-11_morten-chi-squared-cross-reference/winstonlist.dat'
output_csv = 'fit_results.csv'
log_file = 'fit_progress.log'

# Setup logging to file
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def parse_line(line):
    parts = line.strip().split()
    parts = [float(x) for x in parts]
    f162m, err_f162m = parts[2], parts[3]
    f182m, err_f182m = parts[4], parts[5]
    f200w, err_f200w = parts[6], parts[7]
    av = parts[8]
    mass = parts[9]
    mags = [f162m, f182m, f200w]
    errs = [err_f162m, err_f182m, err_f200w]
    return mags, errs, av, mass

def main():
    # Load all lines once
    with open(dat_path, 'r') as f:
        all_lines = f.readlines()

    # Read processed indices if output file exists
    processed_indices = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_indices.add(int(row['index']))
        logging.info(f"Loaded {len(processed_indices)} processed indices from {output_csv}")
    else:
        logging.info("No existing results file found, starting fresh.")

    fitter = ChiSquaredFitter(output_dir='./fit_plots_test')

    write_header = not os.path.exists(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['index', 'best_mass', 'best_AV', 'mass_min', 'mass_max', 'AV_min', 'AV_max', 'intersects', 'min_chi2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total_lines = len(all_lines)
        for idx, line in enumerate(all_lines):
            if idx in processed_indices:
                logging.info(f"Skipping line {idx} (already processed)")
                continue

            try:
                mags, errs, av_true, mass_true = parse_line(line)
                logging.info(f"Processing line {idx+1}/{total_lines} ...")
                result = fitter.analyze_line(idx, mags, errs, av_true, mass_true)

                writer.writerow({
                    'index': result['index'],
                    'best_mass': result['best_mass'],
                    'best_AV': result['best_AV'],
                    'mass_min': result['mass_min'],
                    'mass_max': result['mass_max'],
                    'AV_min': result['AV_min'],
                    'AV_max': result['AV_max'],
                    'intersects': result['intersects'],
                    'min_chi2': result['min_chi2']
                })
                csvfile.flush()

            except Exception as e:
                logging.error(f"Error processing line {idx}: {e}", exc_info=True)

    logging.info("Processing complete.")

if __name__ == '__main__':
    main()