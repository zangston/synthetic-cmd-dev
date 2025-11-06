import os
import numpy as np
import matplotlib.pyplot as plt
from spisea import synthetic, evolution, atmospheres, reddening
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

class ChiSquaredFitter:
    def __init__(self, filt_list, filters, base_iso_dir='isochrones', dist=4500, metallicity=0, log_age=6.0, output_dir='fit_plots'):
        # Separate cache dirs so JWST-only and JWST+HST runs don’t collide
        if len(filt_list) == 3:
            self.iso_dir = os.path.join(base_iso_dir, "jwst_only")
        else:
            self.iso_dir = os.path.join(base_iso_dir, "jwst_hst_f139")
        os.makedirs(self.iso_dir, exist_ok=True)

        self.dist = dist
        self.metallicity = metallicity
        self.log_age = log_age
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.evo_model = evolution.Baraffe15()
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawCardelli(3.1)

        self.filt_list = filt_list
        self.filters = filters

        # JWST + HST F139M filter wavelengths
        self.filter_wavelengths = {
            "m_jwst_F162M": 1.62,
            "m_jwst_F182M": 1.82,
            "m_jwst_F200W": 2.00,
            "m_hst_f139m": 1.39,
        }

        self.AV_to_AKs = 1.0 / 0.1179

    def interpolate_isochrone(self, isochrone, num_interp=10):
        masses = isochrone.points['mass']
        interp_data = []
        for i in range(len(masses) - 1):
            mass1, mass2 = masses[i], masses[i+1]
            interp_masses = np.linspace(mass1, mass2, num=num_interp+2)[1:-1]
            interp_row = {f: np.linspace(isochrone.points[f][i], isochrone.points[f][i+1],
                                         num=num_interp+2)[1:-1] for f in self.filters}
            for j in range(len(interp_masses)):
                entry = [interp_masses[j]] + [interp_row[f][j] for f in self.filters]
                interp_data.append(tuple(entry))
        for i in range(len(masses)):
            entry = [masses[i]] + [isochrone.points[f][i] for f in self.filters]
            interp_data.append(tuple(entry))
        dtype = [('mass', float)] + [(f, float) for f in self.filters]
        return np.array(sorted(interp_data, key=lambda x: x[0]), dtype=dtype)

    def apply_dereddening(self, mags, AKs):
        return [m - self.red_law.Cardelli89(self.filter_wavelengths[self.filters[i]], AKs)
                for i, m in enumerate(mags)]

    def compute_squared_distance(self, dered_mags, interp_table):
        color_obs = dered_mags[0] - dered_mags[1]
        mag_obs = dered_mags[1]
        color_iso = interp_table[self.filters[0]] - interp_table[self.filters[1]]
        mag_iso = interp_table[self.filters[1]]
        distances_squared = (color_iso - color_obs) ** 2 + (mag_iso - mag_obs) ** 2
        return np.min(distances_squared)

    def minimize_AKs(self, mags, interp_table):
        def objective(aks_trial):
            dered = self.apply_dereddening(mags, float(aks_trial))
            return self.compute_squared_distance(dered, interp_table)
        res = minimize_scalar(objective, bounds=(0.0, 3.0), method='bounded')
        return float(res.x)

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
                chi_sq = sum(((mags[i] - model_mags[i])**2) / max(errs[i]**2, 1e-3)
                             for i in range(len(mags)))
                results.append({"AV": av, "AKs": aks, "mass": entry['mass'], "chi2": chi_sq})
        return results

    def analyze_line(self, index, mags, errs, true_av, true_mass):
        iso_unredd = synthetic.IsochronePhot(self.log_age, AKs=0.0, distance=self.dist,
                                             metallicity=self.metallicity,
                                             evo_model=self.evo_model, atm_func=self.atm_func,
                                             red_law=self.red_law, filters=self.filt_list,
                                             iso_dir=self.iso_dir)
        interp_table = self.interpolate_isochrone(iso_unredd, num_interp=10)
        AKs_val = self.minimize_AKs(mags, interp_table)
        AV_val = AKs_val * self.AV_to_AKs
        AV_grid = np.arange(max(0.0, AV_val - 5.0), AV_val + 5.0 + 0.001, 0.1)
        grid_results = self.chi_squared_grid(AV_grid, mags, errs)
        best_fit = min(grid_results, key=lambda x: x["chi2"])
        dof = len(mags) - 2
        chi2_threshold = chi2.ppf(0.997, df=dof)
        acceptable_results = [r for r in grid_results if r['chi2'] <= chi2_threshold]
        if not acceptable_results:
            return {'index': index, 'best_mass': best_fit['mass'], 'best_AV': best_fit['AV'],
                    'mass_min': None, 'mass_max': None, 'AV_min': None, 'AV_max': None,
                    'intersects': None, 'min_chi2': best_fit['chi2']}
        masses = [r['mass'] for r in acceptable_results]
        AVs = [r['AV'] for r in acceptable_results]
        mass_min, mass_max = min(masses), max(masses)
        AV_min, AV_max = min(AVs), max(AVs)
        intersects = (AV_min <= true_av <= AV_max) and (mass_min <= true_mass <= mass_max)
        return {'index': index, 'best_mass': best_fit['mass'], 'best_AV': best_fit['AV'],
                'mass_min': mass_min, 'mass_max': mass_max,
                'AV_min': AV_min, 'AV_max': AV_max,
                'intersects': intersects, 'min_chi2': best_fit['chi2']}