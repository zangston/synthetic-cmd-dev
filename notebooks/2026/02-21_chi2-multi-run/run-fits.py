#!/usr/bin/env python3
"""
run_all_fits.py

Runs a sequence of chi-squared fitting experiments on a single input .dat file,
with per-run:
  - checkpointed CSV output (append-only)
  - per-run log file
  - summary plots:
      1) Mass: identity plot vs Andersen(2024) mass (= true_mass in .dat) + error histogram
      2) AV:   identity plot vs Andersen(2024) AV   (= true_AV   in .dat) + error histogram
      3) IMF: counts + dN/dM (+ power-law slope) + dN/dlogM (+ log-normal fits)

Planned runs:
  - jwst_cmd_F162MminusF182M_vs_F182M  (A1): CMD fit using (F162M-F182M) vs F182M
  - jwst_mags_F162M_F182M              (B1): Two magnitudes-only using F162M & F182M
  - jwst_mags_F162M_F182M_F200W        (C1): Three magnitudes-only using F162M, F182M, F200W
  - jwst_mags_all                      (bucket): same as C1 but separate output folder
  - jwst_hst_mags_all                  (JWST+HST): six magnitudes-only JWST+HST

Key statistical change (important):
- For ALL runs, parameter "acceptance region" (mass_min/mass_max/AV_min/AV_max) is defined via
      chi2 <= chi2_min + Δchi2
  where Δchi2 comes from chi-square CDF with df=2 parameters (mass, AV) at conf (default 0.997).
  This is valid even for 2-observable fits (A1/B1).

- Goodness-of-fit (GOF) hypothesis test is ONLY defined when dof_gof = Nobs - Nparams >= 1.
  We store:
      dof_gof, p_value, passes_gof
  For A1 and B1, dof_gof=0 -> p_value and passes_gof are None.

Plotting change (requested):
- For 2-observable runs (A1/B1), we DO NOT make an "acceptance-only" IMF curve, because
  passes_gof is undefined. Those runs show only the "all best-fits" IMF.

Speed / caching change (important):
- We generate isochrones ONLY for the 6-filter superset (JWST+HST) and reuse the interpolated
  tables for all runs by slicing/filtering in the chi2 calculation.
  This avoids recomputing SPISEA isochrones per run.

Usage:
  python run_all_fits.py \
      --dat /scratch/wyz5rge/synthetic-hr/data/phot_joined_avmass.dat \
      --out_root results_stepwise \
      --log_age 6.0 --metallicity 0 --dist 4500
"""

import os
import csv
import math
import logging
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.optimize import minimize_scalar

from spisea import synthetic, evolution, atmospheres, reddening


# -----------------------------
# Configuration data structure
# -----------------------------
@dataclass(frozen=True)
class FitRunConfig:
    name: str
    filt_list: List[str]          # SPISEA filter list (informational; we run superset internally)
    filters: List[str]            # Isochrone points column names (for mags-only)
    mode: str                     # "mags" or "cmd_a1"


# -----------------------------
# Data parsing + skip predicate
# -----------------------------
def parse_phot_line(parts: List[float]) -> Dict[str, float]:
    """
    Expected columns:
      0 x
      1 y
      2 F162M  3 err162
      4 F182M  5 err182
      6 F200W  7 err200
      8 F125W  9 err125
     10 F139M 11 err139
     12 F160W 13 err160
     14 true_AV
     15 true_mass
    """
    return {
        "x": parts[0],
        "y": parts[1],
        "F162M": parts[2],
        "e162": parts[3],
        "F182M": parts[4],
        "e182": parts[5],
        "F200W": parts[6],
        "e200": parts[7],
        "F125W": parts[8],
        "e125": parts[9],
        "F139M": parts[10],
        "e139": parts[11],
        "F160W": parts[12],
        "e160": parts[13],
        "true_AV": parts[14],
        "true_mass": parts[15],
    }


def should_skip_runner_style(mags: List[float], errs: List[float], true_av: float, true_mass: float) -> bool:
    if true_av <= 0 or true_mass <= 0:
        return True
    for m, e in zip(mags, errs):
        if (m > 90) or (e > 90) or (e <= 0):
            return True
    return False


# -----------------------------
# Plotting + IMF helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fixed_axis_limits_from_truth(
    truth_mass: np.ndarray,
    truth_av: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    m = truth_mass[np.isfinite(truth_mass) & (truth_mass > 0)]
    a = truth_av[np.isfinite(truth_av) & (truth_av > 0)]
    if m.size == 0 or a.size == 0:
        return (0.01, 1.0), (0.0, 50.0)

    m_lo, m_hi = float(np.min(m)), float(np.max(m))
    a_lo, a_hi = float(np.min(a)), float(np.max(a))
    m_lo = max(0.005, m_lo * 0.9)
    m_hi = m_hi * 1.1
    a_lo = max(0.0, a_lo - 0.5)
    a_hi = a_hi + 0.5
    return (m_lo, m_hi), (a_lo, a_hi)


def plot_identity_and_error_hist(
    df: pd.DataFrame,
    out_dir: str,
    run_label: str,
    mass_xlim: Tuple[float, float],
    av_xlim: Tuple[float, float],
) -> None:
    ensure_dir(out_dir)

    df_use = df.copy()
    df_use = df_use[np.isfinite(df_use["true_mass"]) & np.isfinite(df_use["true_AV"])]
    df_use = df_use[(df_use["true_mass"] > 0) & (df_use["true_AV"] > 0)]
    df_use = df_use[np.isfinite(df_use["best_mass"]) & np.isfinite(df_use["best_AV"])]

    df_use["mass_pct_error"] = 100.0 * (df_use["best_mass"] - df_use["true_mass"]) / df_use["true_mass"]
    df_use["av_pct_error"] = 100.0 * (df_use["best_AV"] - df_use["true_AV"]) / df_use["true_AV"]

    # GOF acceptance mask (only meaningful when dof_gof>=1; otherwise passes_gof is None/NaN)
    df_use["has_accept_region"] = df_use["passes_gof"] == True  # noqa: E712

    s = 6
    alpha = 0.35

    acc = df_use[df_use["has_accept_region"]]
    noacc = df_use[~df_use["has_accept_region"]]

    # --- Mass identity ---
    plt.figure(figsize=(7, 7))
    plt.scatter(noacc["true_mass"], noacc["best_mass"], s=s, alpha=alpha, label="fails GOF / undefined", marker="o")
    plt.scatter(acc["true_mass"], acc["best_mass"], s=s, alpha=alpha, label="passes GOF", marker="o")
    plt.plot([mass_xlim[0], mass_xlim[1]], [mass_xlim[0], mass_xlim[1]], "k--", lw=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(mass_xlim)
    plt.ylim(mass_xlim)
    plt.xlabel("Andersen (2024) Mass (M$_\\odot$)")
    plt.ylabel(f"{run_label} Mass (M$_\\odot$)")
    plt.title(f"Mass identity: {run_label} vs Andersen (2024)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mass_identity.png"), dpi=200)
    plt.close()

    # --- Mass error histogram ---
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(-200, 201, 10)
    plt.hist(df_use["mass_pct_error"].dropna(), bins=bins, alpha=0.8)
    plt.xlim(-200, 200)
    plt.xlabel("Mass % Error = 100*(fit - Andersen)/Andersen")
    plt.ylabel("Count")
    plt.title(f"Mass % Error histogram: {run_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mass_error_hist.png"), dpi=200)
    plt.close()

    # --- AV identity ---
    plt.figure(figsize=(7, 7))
    plt.scatter(noacc["true_AV"], noacc["best_AV"], s=s, alpha=alpha, label="fails GOF / undefined", marker="o")
    plt.scatter(acc["true_AV"], acc["best_AV"], s=s, alpha=alpha, label="passes GOF", marker="o")
    plt.plot([av_xlim[0], av_xlim[1]], [av_xlim[0], av_xlim[1]], "k--", lw=1)
    plt.xlim(av_xlim)
    plt.ylim(av_xlim)
    plt.xlabel("Andersen (2024) AV")
    plt.ylabel(f"{run_label} AV")
    plt.title(f"AV identity: {run_label} vs Andersen (2024)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "av_identity.png"), dpi=200)
    plt.close()

    # --- AV error histogram ---
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(-200, 201, 10)
    plt.hist(df_use["av_pct_error"].dropna(), bins=bins, alpha=0.8)
    plt.xlim(-200, 200)
    plt.xlabel("AV % Error = 100*(fit - Andersen)/Andersen")
    plt.ylabel("Count")
    plt.title(f"AV % Error histogram: {run_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "av_error_hist.png"), dpi=200)
    plt.close()


def imf_hist_dndm(masses: np.ndarray, nbins: int = 25):
    masses = np.asarray(masses, dtype=float)
    masses = masses[np.isfinite(masses) & (masses > 0)]
    if masses.size == 0:
        return None

    m_min = max(0.01, float(np.min(masses)))
    m_max = float(np.max(masses))

    edges = np.logspace(np.log10(m_min), np.log10(m_max), nbins + 1)
    N, _ = np.histogram(masses, bins=edges)

    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    dndm = N / widths
    return centers, edges, N, dndm


def fit_powerlaw_slope(centers: np.ndarray, dndm: np.ndarray):
    mask = (dndm > 0) & np.isfinite(dndm) & np.isfinite(centers) & (centers > 0)
    if np.sum(mask) < 3:
        return None
    logM = np.log10(centers[mask])
    logN = np.log10(dndm[mask])
    slope, intercept = np.polyfit(logM, logN, 1)
    alpha = -slope
    return alpha, slope, intercept


def lognormal_dndlogM(M, A, log10_mc, sigma_dex):
    logM = np.log10(M)
    return A * np.exp(-0.5 * ((logM - log10_mc) / sigma_dex) ** 2)


def make_dndlogM(masses: np.ndarray, bin_width_dex: float = 0.1):
    masses = np.asarray(masses, dtype=float)
    masses = masses[np.isfinite(masses) & (masses > 0)]
    if masses.size == 0:
        return None

    logM = np.log10(masses)
    lo = np.floor(logM.min() / bin_width_dex) * bin_width_dex
    hi = np.ceil(logM.max() / bin_width_dex) * bin_width_dex
    edges_log = np.arange(lo, hi + bin_width_dex, bin_width_dex)

    N, _ = np.histogram(logM, bins=edges_log)
    centers_log = 0.5 * (edges_log[:-1] + edges_log[1:])
    centers_M = 10 ** centers_log
    dndlog = N / bin_width_dex
    return centers_M, dndlog, N, edges_log


def fit_lognormal(centers_M: np.ndarray, dndlog: np.ndarray):
    from scipy.optimize import curve_fit

    mask = (dndlog > 0) & np.isfinite(dndlog) & np.isfinite(centers_M) & (centers_M > 0)
    x = centers_M[mask]
    y = dndlog[mask]
    if x.size < 3:
        return None

    A0 = float(np.nanmax(y))
    logmc0 = float(np.log10(x[np.nanargmax(y)]))
    sigma0 = 0.3
    p0 = [A0, logmc0, sigma0]
    bounds = ([0.0, -5.0, 1e-3], [np.inf, 5.0, 2.0])

    popt, pcov = curve_fit(lognormal_dndlogM, x, y, p0=p0, bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_imf_and_fits(
    df: pd.DataFrame,
    out_dir: str,
    run_label: str,
    nbins_logM: int = 25,
    bin_width_dex: float = 0.10,
) -> None:
    """
    Produces:
      - imf_counts.png
      - imf_dndm.png
      - imf_dndlogM_lognormal.png

    Samples:
      * all best fits (always)
      * passes_gof only (only when dof_gof>=1 produces True/False; for dof_gof==0 we skip)
    """
    ensure_dir(out_dir)

    df_use = df.copy()
    df_use = df_use.dropna(subset=["best_mass"])
    df_use = df_use[np.isfinite(df_use["best_mass"])]
    df_use = df_use[df_use["best_mass"] > 0].copy()

    m_all = df_use["best_mass"].to_numpy()

    # GOF-based mask (may be all False/NaN for A1/B1)
    df_use["passes_gof_bool"] = df_use["passes_gof"] == True  # noqa: E712
    m_acc = df_use.loc[df_use["passes_gof_bool"], "best_mass"].to_numpy()

    has_accept_curve = (m_acc.size > 0)

    imf_all = imf_hist_dndm(m_all, nbins=nbins_logM)
    imf_acc = imf_hist_dndm(m_acc, nbins=nbins_logM) if has_accept_curve else None
    if imf_all is None:
        return

    c_all, edges_all, N_all, dndm_all = imf_all
    if imf_acc:
        c_acc, edges_acc, N_acc, dndm_acc = imf_acc
    else:
        c_acc, N_acc, dndm_acc = None, None, None

    fit_all = fit_powerlaw_slope(c_all, dndm_all)
    fit_acc = fit_powerlaw_slope(c_acc, dndm_acc) if imf_acc else None

    # --- Plot counts ---
    plt.figure(figsize=(10, 5))
    plt.step(c_all, N_all, where="mid", color="black", linewidth=2, label=f"{run_label} (all best-fits)")
    if has_accept_curve:
        plt.step(c_acc, N_acc, where="mid", color="crimson", linewidth=2, linestyle="--",
                 label=f"{run_label} (passes GOF only)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("Counts per bin")
    plt.title(f"IMF Histogram (Counts): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_counts.png"), dpi=200)
    plt.close()

    # --- Plot dN/dM + power-law overlay ---
    plt.figure(figsize=(10, 6))
    plt.step(c_all, dndm_all, where="mid", color="black", linewidth=2, label=f"{run_label} dN/dM (all)")
    if has_accept_curve:
        plt.step(c_acc, dndm_acc, where="mid", color="crimson", linewidth=2, linestyle="--",
                 label=f"{run_label} dN/dM (passes GOF)")

    def overlay_powerlaw(fit, mmin, mmax, color, linestyle, label):
        if not fit:
            return
        alpha, slope, intercept = fit
        M = np.logspace(np.log10(mmin), np.log10(mmax), 300)
        y = 10 ** (intercept + slope * np.log10(M))
        plt.plot(M, y, color=color, linestyle=linestyle, linewidth=2, alpha=0.85, label=label)

    mmin = float(np.min(c_all))
    mmax = float(np.max(c_all))
    if fit_all:
        overlay_powerlaw(fit_all, mmin, mmax, "black", "-", f"Power-law fit (all): α={fit_all[0]:.2f}")
    if fit_acc:
        overlay_powerlaw(fit_acc, mmin, mmax, "crimson", "--", f"Power-law fit (passes GOF): α={fit_acc[0]:.2f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("dN/dM")
    plt.title(f"IMF (dN/dM): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_dndm.png"), dpi=200)
    plt.close()

    # --- Log-normal in dN/dlog10M ---
    b_all = make_dndlogM(m_all, bin_width_dex=bin_width_dex)
    b_acc = make_dndlogM(m_acc, bin_width_dex=bin_width_dex) if has_accept_curve else None

    fit_ln_all = fit_lognormal(b_all[0], b_all[1]) if b_all else None
    fit_ln_acc = fit_lognormal(b_acc[0], b_acc[1]) if b_acc else None

    plt.figure(figsize=(10, 7))
    plt.scatter(b_all[0], b_all[1], s=25, color="black", alpha=0.7, label=f"{run_label} dN/dlogM (all)")
    if has_accept_curve and b_acc:
        plt.scatter(b_acc[0], b_acc[1], s=25, color="crimson", alpha=0.7, label=f"{run_label} dN/dlogM (passes GOF)")

    M_lo = float(np.min(b_all[0]))
    M_hi = float(np.max(b_all[0]))
    M_plot = np.logspace(np.log10(M_lo), np.log10(M_hi), 500)

    def overlay_lognormal(fit, color, linestyle, label):
        if not fit:
            return
        (A, logmc, sig), _ = fit
        plt.plot(M_plot, lognormal_dndlogM(M_plot, A, logmc, sig),
                 color=color, linestyle=linestyle, linewidth=2, alpha=0.9, label=label)

    if fit_ln_all:
        (A, logmc, sig), _ = fit_ln_all
        overlay_lognormal(fit_ln_all, "black", "-", f"Log-normal (all): mc={10**logmc:.2f}, σ={sig:.2f}")
    if fit_ln_acc:
        (A, logmc, sig), _ = fit_ln_acc
        overlay_lognormal(fit_ln_acc, "crimson", "--", f"Log-normal (passes GOF): mc={10**logmc:.2f}, σ={sig:.2f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("dN/dlog$_{10}$M")
    plt.title(f"Log-normal IMF fits (bin width = {bin_width_dex:.2f} dex): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_dndlogM_lognormal.png"), dpi=200)
    plt.close()

    # Text summary
    summary_path = os.path.join(out_dir, "imf_fit_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run: {run_label}\n")
        f.write(f"Rows (all best-fits): {len(m_all)}\n")
        if has_accept_curve:
            f.write(f"Rows (passes GOF):    {len(m_acc)}\n")
        else:
            f.write("Rows (passes GOF):    N/A (GOF undefined or none passed)\n")
        f.write("\n")

        if fit_all:
            f.write(f"Power-law (all): alpha={fit_all[0]:.6f}\n")
        else:
            f.write("Power-law (all): not enough bins\n")

        if fit_acc:
            f.write(f"Power-law (passes GOF): alpha={fit_acc[0]:.6f}\n")
        else:
            f.write("Power-law (passes GOF): N/A\n")

        f.write("\n")
        if fit_ln_all:
            (A, logmc, sig), _ = fit_ln_all
            f.write(f"Log-normal (all): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Log-normal (all): not enough points\n")

        if fit_ln_acc:
            (A, logmc, sig), _ = fit_ln_acc
            f.write(f"Log-normal (passes GOF): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Log-normal (passes GOF): N/A\n")


# -----------------------------
# Superset-grid fitter (reuse JWST+HST iso for all runs)
# -----------------------------
class SupersetGridFitter:
    """
    Generates a single (AV, mass) grid using a 6-filter superset isochrone,
    then evaluates chi2 for each run by slicing that same model grid.

    Observed mags/errs are provided per run.
    """

    # Superset filters and their isochrone column keys
    SUPER_FILT_LIST = [
        "jwst,F162M", "jwst,F182M", "jwst,F200W",
        "wfc3,ir,f125w", "wfc3,ir,f139m", "wfc3,ir,f160w",
    ]
    SUPER_FILTERS = [
        "m_jwst_F162M", "m_jwst_F182M", "m_jwst_F200W",
        "m_hst_f125w", "m_hst_f139m", "m_hst_f160w",
    ]
    A1_FILTERS = ["m_jwst_F162M", "m_jwst_F182M"]

    def __init__(
        self,
        iso_dir: str,
        dist: float,
        metallicity: float,
        log_age: float,
        interp_n: int = 10,
        av_grid_halfwidth: float = 5.0,
        av_grid_step: float = 0.1,
        conf: float = 0.997,
    ):
        self.iso_dir = iso_dir
        self.dist = dist
        self.metallicity = metallicity
        self.log_age = log_age
        self.interp_n = interp_n
        self.av_grid_halfwidth = av_grid_halfwidth
        self.av_grid_step = av_grid_step
        self.conf = conf

        ensure_dir(self.iso_dir)

        self.evo_model = evolution.Baraffe15()
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawCardelli(3.1)

        # microns
        self.filter_wavelengths = {
            "m_jwst_F162M": 1.62,
            "m_jwst_F182M": 1.82,
            "m_jwst_F200W": 2.00,
            "m_hst_f125w": 1.25,
            "m_hst_f139m": 1.39,
            "m_hst_f160w": 1.60,
        }

        self.AV_to_AKs = 1.0 / 0.1179
        self.AKs_per_AV = 0.118

        # Δχ² for df=2 parameters (mass, AV)
        self.delta_chi2 = float(chi2.ppf(self.conf, df=2))

    @staticmethod
    def _cmd_obs(m162: float, m182: float) -> Tuple[float, float]:
        return (m162 - m182, m182)

    @staticmethod
    def _cmd_sigma(e162: float, e182: float) -> Tuple[float, float]:
        return (math.sqrt(e162 * e162 + e182 * e182), e182)

    def interpolate_isochrone_superset(self, isochrone, num_interp: int) -> np.ndarray:
        masses = isochrone.points["mass"]
        interp_data = []

        use_filters = self.SUPER_FILTERS

        for i in range(len(masses) - 1):
            m1, m2 = masses[i], masses[i + 1]
            interp_masses = np.linspace(m1, m2, num=num_interp + 2)[1:-1]

            interp_row = {
                f: np.linspace(isochrone.points[f][i], isochrone.points[f][i + 1], num=num_interp + 2)[1:-1]
                for f in use_filters
            }

            for j in range(len(interp_masses)):
                entry = [interp_masses[j]] + [interp_row[f][j] for f in use_filters]
                interp_data.append(tuple(entry))

        for i in range(len(masses)):
            entry = [masses[i]] + [isochrone.points[f][i] for f in use_filters]
            interp_data.append(tuple(entry))

        dtype = [("mass", float)] + [(f, float) for f in use_filters]
        return np.array(sorted(interp_data, key=lambda x: x[0]), dtype=dtype)

    def apply_dereddening(self, mags: List[float], AKs: float, filters: List[str]) -> List[float]:
        out = []
        for i, m in enumerate(mags):
            wl = self.filter_wavelengths[filters[i]]
            out.append(m - self.red_law.Cardelli89(wl, AKs))
        return out

    def minimize_AKs_from_162_182(self, m162: float, e162: float, m182: float, e182: float) -> float:
        """
        Initial guess for AV via minimizing CMD distance on unreddened *superset* isochrone.
        """
        iso_unredd = synthetic.IsochronePhot(
            self.log_age,
            AKs=0.0,
            distance=self.dist,
            metallicity=self.metallicity,
            evo_model=self.evo_model,
            atm_func=self.atm_func,
            red_law=self.red_law,
            filters=self.SUPER_FILT_LIST,
            iso_dir=self.iso_dir,
        )
        interp = self.interpolate_isochrone_superset(iso_unredd, num_interp=self.interp_n)

        def objective(aks_trial: float) -> float:
            aks_trial = float(aks_trial)
            dered = self.apply_dereddening([m162, m182], aks_trial, self.A1_FILTERS)

            c_obs, y_obs = self._cmd_obs(dered[0], dered[1])
            c_iso = interp[self.A1_FILTERS[0]] - interp[self.A1_FILTERS[1]]
            y_iso = interp[self.A1_FILTERS[1]]
            d2 = (c_iso - c_obs) ** 2 + (y_iso - y_obs) ** 2
            return float(np.min(d2))

        res = minimize_scalar(objective, bounds=(0.0, 3.0), method="bounded")
        return float(res.x)

    def chi2_for_run_on_entry(
        self,
        cfg: FitRunConfig,
        entry: np.void,
        obs_mags: List[float],
        obs_errs: List[float],
    ) -> float:
        if cfg.mode == "cmd_a1":
            m162_obs, m182_obs = obs_mags[0], obs_mags[1]
            e162, e182 = obs_errs[0], obs_errs[1]

            c_obs, y_obs = self._cmd_obs(m162_obs, m182_obs)
            sig_c, sig_y = self._cmd_sigma(e162, e182)

            m162_mod = float(entry["m_jwst_F162M"])
            m182_mod = float(entry["m_jwst_F182M"])
            c_mod, y_mod = self._cmd_obs(m162_mod, m182_mod)

            sig_c = max(sig_c, 1e-3)
            sig_y = max(sig_y, 1e-3)

            return float(((c_obs - c_mod) ** 2) / (sig_c * sig_c) + ((y_obs - y_mod) ** 2) / (sig_y * sig_y))

        # mags-only
        chi2_val = 0.0
        for i, f in enumerate(cfg.filters):
            sig = max(float(obs_errs[i]), 1e-3)
            diff = float(obs_mags[i]) - float(entry[f])
            chi2_val += (diff * diff) / (sig * sig)
        return float(chi2_val)

    def analyze_star_for_runs(
        self,
        index_noncomment: int,
        file_lineno: int,
        row: Dict[str, float],
        runs_needed: List[FitRunConfig],
        filt_to_dat: Dict[str, Tuple[str, str]],
    ) -> Dict[str, Dict[str, object]]:
        """
        Compute results for only the runs in runs_needed for this star.
        Returns dict: run_name -> result row dict.
        """

        # Observations needed for AV guess (always from 162/182)
        m162, e162 = row["F162M"], row["e162"]
        m182, e182 = row["F182M"], row["e182"]

        # If AV guess inputs fail runner predicate, we will likely skip everything anyway.
        aks_guess = self.minimize_AKs_from_162_182(m162, e162, m182, e182)
        av_guess = aks_guess * self.AV_to_AKs

        av_lo = max(0.0, av_guess - self.av_grid_halfwidth)
        av_hi = av_guess + self.av_grid_halfwidth
        av_grid = np.arange(av_lo, av_hi + 1e-9, self.av_grid_step)

        # Build obs vectors per run
        obs_by_run: Dict[str, Tuple[List[float], List[float], int]] = {}
        for cfg in runs_needed:
            if cfg.mode == "cmd_a1":
                obs_mags = [row["F162M"], row["F182M"]]
                obs_errs = [row["e162"], row["e182"]]
                nobs = 2
            else:
                obs_mags = []
                obs_errs = []
                for f_iso in cfg.filters:
                    m_key, e_key = filt_to_dat[f_iso]
                    obs_mags.append(row[m_key])
                    obs_errs.append(row[e_key])
                nobs = len(obs_mags)

            obs_by_run[cfg.name] = (obs_mags, obs_errs, nobs)

        # Grid search (single superset model grid reused for all runs)
        # We'll store (AV, AKs, mass, model_mags...) implicitly via entry + av.
        results_by_run: Dict[str, List[Tuple[float, float, float, float]]] = {cfg.name: [] for cfg in runs_needed}
        # tuple: (AV, AKs, mass, chi2_run)

        for av in av_grid:
            aks = av * self.AKs_per_AV
            iso = synthetic.IsochronePhot(
                self.log_age,
                AKs=aks,
                distance=self.dist,
                metallicity=self.metallicity,
                evo_model=self.evo_model,
                atm_func=self.atm_func,
                red_law=self.red_law,
                filters=self.SUPER_FILT_LIST,
                iso_dir=self.iso_dir,
            )
            interp = self.interpolate_isochrone_superset(iso, num_interp=self.interp_n)

            for entry in interp:
                mass_val = float(entry["mass"])
                for cfg in runs_needed:
                    obs_mags, obs_errs, _ = obs_by_run[cfg.name]
                    chi2_val = self.chi2_for_run_on_entry(cfg, entry, obs_mags, obs_errs)
                    results_by_run[cfg.name].append((float(av), float(aks), mass_val, float(chi2_val)))

        out: Dict[str, Dict[str, object]] = {}
        for cfg in runs_needed:
            arr = np.array(
                results_by_run[cfg.name],
                dtype=[("AV", float), ("AKs", float), ("mass", float), ("chi2", float)],
            )
            if arr.size == 0 or not np.any(np.isfinite(arr["chi2"])):
                raise RuntimeError(f"Run {cfg.name}: grid produced no finite chi2 values.")

            best_idx = int(np.nanargmin(arr["chi2"]))
            best = arr[best_idx]
            min_chi2 = float(best["chi2"])

            chi2_threshold = min_chi2 + self.delta_chi2
            acceptable = arr[np.isfinite(arr["chi2"]) & (arr["chi2"] <= chi2_threshold)]

            # GOF (only when dof_gof>=1)
            _, _, nobs = obs_by_run[cfg.name]
            dof_gof = int(nobs - 2)
            passes_gof = None
            p_value = None
            if dof_gof >= 1:
                p_value = float(chi2.sf(min_chi2, df=dof_gof))
                passes_gof = bool(p_value > (1.0 - self.conf))

            # Bounds
            if acceptable.size == 0:
                mass_min = mass_max = av_min = av_max = None
                intersects = None
            else:
                mass_min = float(np.min(acceptable["mass"]))
                mass_max = float(np.max(acceptable["mass"]))
                av_min = float(np.min(acceptable["AV"]))
                av_max = float(np.max(acceptable["AV"]))
                intersects = (av_min <= row["true_AV"] <= av_max) and (mass_min <= row["true_mass"] <= mass_max)

            out[cfg.name] = {
                "index": index_noncomment,
                "file_lineno": file_lineno,
                "best_mass": float(best["mass"]),
                "best_AV": float(best["AV"]),
                "mass_min": mass_min,
                "mass_max": mass_max,
                "AV_min": av_min,
                "AV_max": av_max,
                "intersects": (bool(intersects) if intersects is not None else None),
                "min_chi2": min_chi2,
                "chi2_threshold": chi2_threshold,
                "dof_gof": dof_gof,
                "passes_gof": passes_gof,
                "p_value": p_value,
            }

        return out


# -----------------------------
# Orchestration
# -----------------------------
def setup_run_io(out_root: str, runs: List[FitRunConfig]) -> Tuple[
    Dict[str, str],
    Dict[str, logging.Logger],
    Dict[str, set],
]:
    """
    Returns:
      csv_path_by_run, logger_by_run, processed_index_set_by_run
    """
    csv_paths: Dict[str, str] = {}
    loggers: Dict[str, logging.Logger] = {}
    processed: Dict[str, set] = {}

    for cfg in runs:
        run_dir = os.path.join(out_root, cfg.name)
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "plots"))

        # Logger
        log_path = os.path.join(run_dir, f"{cfg.name}.log")
        logger = logging.getLogger(cfg.name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
        loggers[cfg.name] = logger

        logger.info("=== INIT RUN %s ===", cfg.name)

        # CSV
        csv_path = os.path.join(run_dir, f"fit_results_{cfg.name}.csv")
        csv_paths[cfg.name] = csv_path

        # Load processed indices
        processed_set = set()
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    for r in csv.DictReader(f):
                        processed_set.add(int(r["index"]))
                logger.info("Loaded %d processed indices from existing CSV", len(processed_set))
            except Exception as e:
                logger.error("Could not read existing CSV for checkpointing: %s", e, exc_info=True)
        processed[cfg.name] = processed_set

        # Ensure header exists
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                fieldnames = [
                    "index",
                    "file_lineno",
                    "best_mass",
                    "best_AV",
                    "mass_min",
                    "mass_max",
                    "AV_min",
                    "AV_max",
                    "intersects",
                    "min_chi2",
                    "chi2_threshold",
                    "dof_gof",
                    "passes_gof",
                    "p_value",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    return csv_paths, loggers, processed


def append_row(csv_path: str, row: Dict[str, object]) -> None:
    fieldnames = [
        "index",
        "file_lineno",
        "best_mass",
        "best_AV",
        "mass_min",
        "mass_max",
        "AV_min",
        "AV_max",
        "intersects",
        "min_chi2",
        "chi2_threshold",
        "dof_gof",
        "passes_gof",
        "p_value",
    ]
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)
        csvfile.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=str, required=True, help="Path to phot_joined_avmass.dat")
    parser.add_argument("--out_root", type=str, default="results_stepwise", help="Output root directory")
    parser.add_argument("--base_iso_dir", type=str, default="isochrones_stepwise", help="Base isochrone cache root")
    parser.add_argument("--dist", type=float, default=4500.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--log_age", type=float, default=6.0)
    parser.add_argument("--interp_n", type=int, default=10)
    parser.add_argument("--av_grid_halfwidth", type=float, default=5.0)
    parser.add_argument("--av_grid_step", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.997)

    parser.add_argument("--nbins_logM", type=int, default=25)
    parser.add_argument("--bin_width_dex", type=float, default=0.10)

    args = parser.parse_args()

    ensure_dir(args.out_root)
    ensure_dir(args.base_iso_dir)

    runs: List[FitRunConfig] = [
        FitRunConfig(
            name="jwst_cmd_F162MminusF182M_vs_F182M",
            filt_list=["jwst,F162M", "jwst,F182M"],
            filters=["m_jwst_F162M", "m_jwst_F182M"],
            mode="cmd_a1",
        ),
        FitRunConfig(
            name="jwst_mags_F162M_F182M",
            filt_list=["jwst,F162M", "jwst,F182M"],
            filters=["m_jwst_F162M", "m_jwst_F182M"],
            mode="mags",
        ),
        FitRunConfig(
            name="jwst_mags_F162M_F182M_F200W",
            filt_list=["jwst,F162M", "jwst,F182M", "jwst,F200W"],
            filters=["m_jwst_F162M", "m_jwst_F182M", "m_jwst_F200W"],
            mode="mags",
        ),
        FitRunConfig(
            name="jwst_mags_all",
            filt_list=["jwst,F162M", "jwst,F182M", "jwst,F200W"],
            filters=["m_jwst_F162M", "m_jwst_F182M", "m_jwst_F200W"],
            mode="mags",
        ),
        FitRunConfig(
            name="jwst_hst_mags_all",
            filt_list=[
                "jwst,F162M", "jwst,F182M", "jwst,F200W",
                "wfc3,ir,f125w", "wfc3,ir,f139m", "wfc3,ir,f160w",
            ],
            filters=[
                "m_jwst_F162M", "m_jwst_F182M", "m_jwst_F200W",
                "m_hst_f125w", "m_hst_f139m", "m_hst_f160w",
            ],
            mode="mags",
        ),
    ]

    # Per-run logs/csvs/checkpoint sets
    csv_paths, loggers, processed_by_run = setup_run_io(args.out_root, runs)

    # Isochrone cache directory (single superset cache shared by all runs)
    superset_iso_dir = os.path.join(args.base_iso_dir, "superset_jwst_hst_all6_cache")
    ensure_dir(superset_iso_dir)

    fitter = SupersetGridFitter(
        iso_dir=superset_iso_dir,
        dist=args.dist,
        metallicity=args.metallicity,
        log_age=args.log_age,
        interp_n=args.interp_n,
        av_grid_halfwidth=args.av_grid_halfwidth,
        av_grid_step=args.av_grid_step,
        conf=args.conf,
    )

    # Mapping for mags-only runs
    filt_to_dat = {
        "m_jwst_F162M": ("F162M", "e162"),
        "m_jwst_F182M": ("F182M", "e182"),
        "m_jwst_F200W": ("F200W", "e200"),
        "m_hst_f125w":  ("F125W", "e125"),
        "m_hst_f139m":  ("F139M", "e139"),
        "m_hst_f160w":  ("F160W", "e160"),
    }

    # Pre-scan truth for later plotting + also allow fast truth-by-index mapping
    truth_by_index: Dict[int, Tuple[float, float]] = {}
    truth_mass_all: List[float] = []
    truth_av_all: List[float] = []

    idx_noncomment = -1
    with open(args.dat, "r") as f:
        for lineno_1based, line in enumerate(f, start=1):
            if line.startswith("#") or (not line.strip()):
                continue
            idx_noncomment += 1
            parts = [float(x) for x in line.split()]
            if len(parts) < 16:
                continue
            row = parse_phot_line(parts)
            truth_by_index[idx_noncomment] = (row["true_AV"], row["true_mass"])
            if row["true_mass"] > 0 and row["true_AV"] > 0:
                truth_mass_all.append(row["true_mass"])
                truth_av_all.append(row["true_AV"])

    # Main pass: iterate file again, compute only missing runs per index
    idx_noncomment = -1
    with open(args.dat, "r") as f:
        for lineno_1based, line in enumerate(f, start=1):
            if line.startswith("#") or (not line.strip()):
                continue
            idx_noncomment += 1

            # Determine which runs still need this index
            runs_needed = [cfg for cfg in runs if idx_noncomment not in processed_by_run[cfg.name]]
            if not runs_needed:
                continue

            parts = [float(x) for x in line.split()]
            if len(parts) < 16:
                for cfg in runs_needed:
                    loggers[cfg.name].info("Skip idx=%d lineno=%d (too few columns: %d)", idx_noncomment, lineno_1based, len(parts))
                continue

            row = parse_phot_line(parts)

            # Apply runner predicate per run, and only keep runs that pass
            runs_ready: List[FitRunConfig] = []
            for cfg in runs_needed:
                if cfg.mode == "cmd_a1":
                    mags = [row["F162M"], row["F182M"]]
                    errs = [row["e162"], row["e182"]]
                else:
                    mags = []
                    errs = []
                    for f_iso in cfg.filters:
                        m_key, e_key = filt_to_dat[f_iso]
                        mags.append(row[m_key])
                        errs.append(row[e_key])

                if should_skip_runner_style(mags, errs, row["true_AV"], row["true_mass"]):
                    loggers[cfg.name].info("Skip idx=%d lineno=%d (runner predicate)", idx_noncomment, lineno_1based)
                    processed_by_run[cfg.name].add(idx_noncomment)  # treat as done (won't ever be fit)
                    continue

                runs_ready.append(cfg)

            if not runs_ready:
                continue

            # Compute fits (shared isochrone work)
            try:
                results = fitter.analyze_star_for_runs(
                    index_noncomment=idx_noncomment,
                    file_lineno=lineno_1based,
                    row=row,
                    runs_needed=runs_ready,
                    filt_to_dat=filt_to_dat,
                )

                for cfg in runs_ready:
                    res = results[cfg.name]
                    append_row(csv_paths[cfg.name], res)
                    processed_by_run[cfg.name].add(idx_noncomment)

                    if (idx_noncomment % 10) == 0:
                        loggers[cfg.name].info(
                            "Processed idx=%d lineno=%d best_mass=%.6f best_AV=%.6f min_chi2=%.4f dof_gof=%d",
                            idx_noncomment, lineno_1based,
                            res["best_mass"], res["best_AV"], res["min_chi2"], res["dof_gof"],
                        )

            except Exception as e:
                for cfg in runs_ready:
                    loggers[cfg.name].error("Error processing idx=%d lineno=%d: %s", idx_noncomment, lineno_1based, e, exc_info=True)

    # Global axis limits from truth
    truth_mass = np.asarray(truth_mass_all, dtype=float)
    truth_av = np.asarray(truth_av_all, dtype=float)
    mass_xlim, av_xlim = fixed_axis_limits_from_truth(truth_mass, truth_av)

    # Post: attach truth + plots for each run
    for cfg in runs:
        run_dir = os.path.join(args.out_root, cfg.name)
        csv_path = csv_paths[cfg.name]
        logger = loggers[cfg.name]
        logger.info("=== POSTPROCESS RUN %s ===", cfg.name)

        if not os.path.exists(csv_path):
            logger.error("Missing CSV: %s", csv_path)
            continue

        try:
            df = pd.read_csv(csv_path)
            df["true_AV"] = df["index"].map(lambda i: truth_by_index.get(int(i), (np.nan, np.nan))[0])
            df["true_mass"] = df["index"].map(lambda i: truth_by_index.get(int(i), (np.nan, np.nan))[1])

            out_truth_csv = os.path.join(run_dir, f"fit_results_{cfg.name}_with_truth.csv")
            df.to_csv(out_truth_csv, index=False)
            logger.info("Wrote with-truth CSV: %s", out_truth_csv)

            # Identity/error plots with fixed limits
            plot_identity_and_error_hist(
                df=df,
                out_dir=os.path.join(run_dir, "compare_identity"),
                run_label=cfg.name,
                mass_xlim=mass_xlim,
                av_xlim=av_xlim,
            )

            # IMF plots (with GOF behavior described in docstring)
            plot_imf_and_fits(
                df,
                out_dir=os.path.join(run_dir, "imf"),
                run_label=cfg.name,
                nbins_logM=args.nbins_logM,
                bin_width_dex=args.bin_width_dex,
            )

        except Exception as e:
            logger.error("Post-run plotting failed: %s", e, exc_info=True)

    # Top-level summary
    summary_path = os.path.join(args.out_root, "RUN_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("Stepwise fitting outputs\n")
        f.write(f"Input dat: {args.dat}\n")
        f.write(f"dist={args.dist} metallicity={args.metallicity} log_age={args.log_age}\n")
        f.write(f"Global mass xlim: {mass_xlim}\n")
        f.write(f"Global AV xlim: {av_xlim}\n")
        f.write(f"Superset iso cache: {superset_iso_dir}\n\n")
        for cfg in runs:
            f.write(f"{cfg.name}:\n")
            f.write(f"  CSV:   {os.path.join(args.out_root, cfg.name, f'fit_results_{cfg.name}.csv')}\n")
            f.write(f"  Truth: {os.path.join(args.out_root, cfg.name, f'fit_results_{cfg.name}_with_truth.csv')}\n")
            f.write(f"  Log:   {os.path.join(args.out_root, cfg.name, f'{cfg.name}.log')}\n")
            f.write(f"  Plots: {os.path.join(args.out_root, cfg.name)}\n\n")

    print(f"Done. See: {summary_path}")


if __name__ == "__main__":
    main()