import os
import numpy as np
from astropy.table import Table, vstack

def generate_log_age_range(start, end, increment):
    """
    Generate an array of log ages from start to end with the given increment.
    """
    return np.arange(start, end + increment, increment)

def fits_to_dat(fits_file, dat_file, mass_limit=0.7):
    """
    Convert FITS file to DAT file, filtering data to include only rows with mass < mass_limit.
    """
    # Read FITS data
    fits_data = Table.read(fits_file, format='fits')

    # Take base-10 logarithm of Teff to get Log T
    fits_data['logT'] = np.log10(fits_data['Teff'])

    # Use logT for logT_WR (assuming low mass stars)
    fits_data['logT_WR'] = fits_data['logT']

    # Use Mass column for M_init and M_curr
    fits_data['M_init'] = fits_data['Mass']
    fits_data['M_curr'] = fits_data['Mass']

    # Set phase to 1 for all rows
    fits_data['phase'] = np.ones(len(fits_data), dtype=int)

    # Filter data to only include rows with mass < mass_limit
    filtered_data = fits_data[fits_data['M_init'] < mass_limit]

    # Write to DAT file with headers
    header = "# M_init       log T       log L       log g    logT_WR       M_curr phase Source    \n"
    header += "# (Msun)    (Kelvin)      (Lsun)       (cgs)   (Kelvin)       (Msun)    () ()        \n"
    lines = [header]
    for row in filtered_data:
        line = f"{row['M_init']:.6f} {row['logT']:.4f} {row['logL']:.4f} {row['logG']:.4f} {row['logT_WR']:.4f} {row['M_curr']:.6f} {row['phase']} Baraffe\n"
        lines.append(line)

    # Write to DAT file
    with open(dat_file, 'w') as f:
        f.writelines(lines)

def prepend_baraffe_data(baraffe_dir, merged_model_dir, new_model_dir, log_age_range):
    # Ensure the new model directory exists
    os.makedirs(new_model_dir, exist_ok=True)

    for log_age in log_age_range:
        # Construct file names
        age_str = f"{log_age:.2f}"
        baraffe_fits_file = os.path.join(baraffe_dir, f"iso_{age_str}.fits")
        baraffe_dat_file = os.path.join(baraffe_dir, f"iso_{age_str}.dat")
        merged_dat_file = os.path.join(merged_model_dir, f"iso_{age_str}.dat")
        new_dat_file = os.path.join(new_model_dir, f"iso_{age_str}.dat")
        merged_fits_file = os.path.join(merged_model_dir, f"iso_{age_str}.fits")
        new_fits_file = os.path.join(new_model_dir, f"iso_{age_str}.fits")
        
        print(f"Processing age {age_str}: DAT files")

        # Convert Baraffe FITS to DAT
        fits_to_dat(baraffe_fits_file, baraffe_dat_file, mass_limit=0.7)
        
        print(f"Generated DAT file: {os.path.abspath(baraffe_dat_file)}")

        # Read existing merged model data (DAT)
        with open(merged_dat_file, 'r') as f:
            merged_dat_content = f.readlines()

        # Read Baraffe DAT content
        with open(baraffe_dat_file, 'r') as f:
            baraffe_dat_content = f.readlines()

        # Prepend Baraffe data to merged model data (DAT)
        new_dat_content = baraffe_dat_content + merged_dat_content

        # Write new DAT file
        with open(new_dat_file, 'w') as f:
            f.writelines(new_dat_content)
        
        print(f"Generated DAT file: {os.path.abspath(new_dat_file)}")

        print(f"Processing age {age_str}: FITS files")

        # Read Baraffe FITS data
        baraffe_fits_data = Table.read(baraffe_fits_file, format='fits')

        # Read existing merged model FITS data
        merged_fits_data = Table.read(merged_fits_file, format='fits')

        # Prepend Baraffe FITS data to merged model FITS data
        new_fits_data = vstack([baraffe_fits_data, merged_fits_data])

        # Write new FITS file
        new_fits_data.write(new_fits_file, format='fits', overwrite=True)
        
        print(f"Generated FITS file: {os.path.abspath(new_fits_file)}")

def run_script(baraffe_path, merged_model_path, new_model_path):
    # Define directories
    baraffe_dir = baraffe_path
    merged_model_dir = merged_model_path
    new_model_dir = new_model_path

    # Generate log age range
    log_age_range = generate_log_age_range(6.00, 8.00, 0.01)

    # Run the function
    prepend_baraffe_data(baraffe_dir, merged_model_dir, new_model_dir, log_age_range)
