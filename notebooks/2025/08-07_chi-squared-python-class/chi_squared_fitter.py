import os
import numpy as np
import matplotlib.pyplot as plt
from spisea import synthetic, evolution, atmospheres, reddening
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

class ChiSquaredFitter:
    def __init__(self, iso_dir='isochrones/', dist=4500, metallicity=0, log_age=6.0, output_dir='fit_plots'):
        self.iso_dir = iso_dir
        self.dist = dist
        self.metallicity = metallicity
        self.log_age = log_age  # log10(age in years)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Models and filters setup
        self.evo_model = evolution.Baraffe15()
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawCardelli(3.1)

        self.filt_list = ['jwst,F162M', 'jwst,F182M', 'jwst,F200W']
        self.filters = ['m_jwst_F162M', 'm_jwst_F182M', 'm_jwst_F200W']

        self.filter_wavelengths = {
            "m_jwst_F162M": 1.62,
            "m_jwst_F182M": 1.82,
            "m_jwst_F200W": 2.00
        }

        self.AV_to_AKs = 1.0 / 0.1179  # ≈ 8.479

    def interpolate_isochrone(self, isochrone, num_interp=10):
        masses = isochrone.points['mass']
        m1 = isochrone.points['m_jwst_F162M']
        m2 = isochrone.points['m_jwst_F182M']
        m3 = isochrone.points['m_jwst_F200W']

        interp_data = []
        for i in range(len(masses) - 1):
            mass1, mass2 = masses[i], masses[i+1]
            m1_1, m1_2 = m1[i], m1[i+1]
            m2_1, m2_2 = m2[i], m2[i+1]
            m3_1, m3_2 = m3[i], m3[i+1]

            interp_masses = np.linspace(mass1, mass2, num=num_interp+2)[1:-1]
            interp_m1 = np.linspace(m1_1, m1_2, num=num_interp+2)[1:-1]
            interp_m2 = np.linspace(m2_1, m2_2, num=num_interp+2)[1:-1]
            interp_m3 = np.linspace(m3_1, m3_2, num=num_interp+2)[1:-1]

            for m, im1, im2, im3 in zip(interp_masses, interp_m1, interp_m2, interp_m3):
                interp_data.append((m, im1, im2, im3))

        for i in range(len(masses)):
            interp_data.append((masses[i], m1[i], m2[i], m3[i]))

        interp_data = sorted(interp_data, key=lambda x: x[0])
        dtype = [('mass', float), ('m_jwst_F162M', float), ('m_jwst_F182M', float), ('m_jwst_F200W', float)]
        return np.array(interp_data, dtype=dtype)

    def apply_dereddening(self, mags, AKs):
        return [
            m - self.red_law.Cardelli89(self.filter_wavelengths[self.filters[i]], AKs)
            for i, m in enumerate(mags)
        ]

    def compute_squared_distance(self, dered_mags, interp_table):
        color_obs = dered_mags[0] - dered_mags[1]
        mag_obs = dered_mags[1]

        color_iso = interp_table['m_jwst_F162M'] - interp_table['m_jwst_F182M']
        mag_iso = interp_table['m_jwst_F182M']

        distances_squared = (color_iso - color_obs) ** 2 + (mag_iso - mag_obs) ** 2
        return np.min(distances_squared)

    def minimize_AKs(self, mags, interp_table):
        def objective(aks_trial):
            aks_trial = float(np.squeeze(aks_trial))
            dered = self.apply_dereddening(mags, aks_trial)
            return self.compute_squared_distance(dered, interp_table)

        res = minimize_scalar(objective, bounds=(0.0, 3.0), method='bounded')
        return float(np.squeeze(res.x))

    def chi_squared_grid(self, AV_grid, mags, errs):
        AKs_per_AV = 0.118
        results = []

        for av in AV_grid:
            aks = av * AKs_per_AV
            iso = synthetic.IsochronePhot(self.log_age, AKs=aks, distance=self.dist,
                                          metallicity=self.metallicity,
                                          evo_model=self.evo_model, atm_func=self.atm_func,
                                          red_law=self.red_law, filters=self.filt_list,
                                          iso_dir=self.iso_dir)
            interp = self.interpolate_isochrone(iso, num_interp=10)
            for entry in interp:
                model_mags = [entry[f] for f in self.filters]
                dof = len(mags) - 2
                chi_sq = sum(((mags[i] - model_mags[i]) ** 2) / max(errs[i] ** 2, 1e-3) for i in range(len(mags)))
                results.append({
                    "AV": av,
                    "AKs": aks,
                    "mass": entry['mass'],
                    "chi2": chi_sq
                })
        return results

    def analyze_line(self, index, mags, errs, true_av, true_mass):
        # Step 1: Generate unreddened isochrone and interpolate
        iso_unredd = synthetic.IsochronePhot(self.log_age, AKs=0.0, distance=self.dist,
                                             metallicity=self.metallicity,
                                             evo_model=self.evo_model, atm_func=self.atm_func,
                                             red_law=self.red_law, filters=self.filt_list,
                                             iso_dir=self.iso_dir)
        interp_table = self.interpolate_isochrone(iso_unredd, num_interp=10)

        # Step 2: Minimize AKs to get good extinction guess
        AKs_val = self.minimize_AKs(mags, interp_table)
        AV_val = AKs_val * self.AV_to_AKs

        # Step 3: Define AV grid for chi-squared grid search
        delta_AV = 5.0
        AV_low = max(0.0, AV_val - delta_AV)
        AV_high = AV_val + delta_AV
        AV_grid = np.arange(AV_low, AV_high + 0.001, 0.1)

        # Step 4: Compute chi-squared grid over AV and mass
        grid_results = self.chi_squared_grid(AV_grid, mags, errs)
        grid_results_sorted = sorted(grid_results, key=lambda x: x["chi2"])

        best_fit = grid_results_sorted[0]
        min_chi2 = best_fit['chi2']

        dof = len(mags) - 2  # = 1 for 3 filters and 2 fitted params
        confidence = 0.997
        chi2_threshold = chi2.ppf(confidence, df=dof)

        # Find acceptable region points
        acceptable_results = [r for r in grid_results_sorted if r['chi2'] <= chi2_threshold]
        acceptable_masses = [r['mass'] for r in acceptable_results]
        acceptable_AVs = [r['AV'] for r in acceptable_results]

        mass_min, mass_max = min(acceptable_masses), max(acceptable_masses)
        AV_min, AV_max = min(acceptable_AVs), max(acceptable_AVs)

        # Check if ground truth is inside rejection region
        intersects = (true_av >= AV_min and true_av <= AV_max and
                      true_mass >= mass_min and true_mass <= mass_max)

        # Plot results
        self.plot_results(index, grid_results, best_fit, true_av, true_mass, chi2_threshold)

        return {
            'index': index,
            'best_mass': best_fit['mass'],
            'best_AV': best_fit['AV'],
            'mass_min': mass_min,
            'mass_max': mass_max,
            'AV_min': AV_min,
            'AV_max': AV_max,
            'intersects': intersects,
            'min_chi2': min_chi2
        }

    def plot_results(self, index, grid_results, best_fit, true_av, true_mass, chi2_threshold):
        import matplotlib.lines as mlines
        AV_vals = sorted(set(r['AV'] for r in grid_results))
        mass_vals = sorted(set(r['mass'] for r in grid_results))
        chi2_grid = np.full((len(mass_vals), len(AV_vals)), np.nan)
        AV_index = {av: i for i, av in enumerate(AV_vals)}
        mass_index = {m: i for i, m in enumerate(mass_vals)}

        for r in grid_results:
            i = mass_index[r['mass']]
            j = AV_index[r['AV']]
            chi2_grid[i, j] = r['chi2']

        AVs_grid_2d, mass_grid_2d = np.meshgrid(AV_vals, mass_vals)

        plt.figure(figsize=(10, 7))
        mask = chi2_grid < chi2_threshold
        sc = plt.scatter(AVs_grid_2d[mask], mass_grid_2d[mask], c=chi2_grid[mask], cmap='seismic', s=10)
        cbar = plt.colorbar(sc)
        cbar.set_label('Reduced χ²')

        plt.scatter([best_fit['AV']], [best_fit['mass']], color='lime', marker='x', s=100, label='Best fit')
        plt.scatter([true_av], [true_mass], color='gold', marker='*', s=150, label='Ground truth')

        crit_label = f'χ² < {chi2_threshold:.2f}'
        crit_handle = mlines.Line2D([], [], color='none', label=crit_label)

        plt.xlabel('AV (Visual Extinction)')
        plt.ylabel('Mass (M☉)')
        plt.title(f'Acceptable Region (χ² < Critical Value) - Line {index}')
        plt.legend(handles=[crit_handle,
                            plt.Line2D([], [], color='lime', marker='x', linestyle='None', markersize=10, label='Best fit'),
                            plt.Line2D([], [], color='gold', marker='*', linestyle='None', markersize=12, label='Ground truth')])
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"fit_{index}.png")
        plt.savefig(plot_path)
        plt.close()
