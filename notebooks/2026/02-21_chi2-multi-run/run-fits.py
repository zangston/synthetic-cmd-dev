#!/usr/bin/env python3
"""
run_all_fits.py

Runs a sequence of chi-squared fitting experiments on a single input .dat file.

Per-run outputs:
  - checkpointed CSV output (append-only)
  - per-run log file
  - plots:
      * compare_identity/: identity + error hist for Mass and AV
      * imf/: matched-sample IMFs (Andersen vs Fit; all vs accept-only)
      * plots/: per-star (AV, mass) chi2/acceptance diagnostic

Key statistical definitions:
- "Acceptance region" is ALWAYS defined in parameter space (mass, AV) via:
      chi2 <= chi2_min + Δchi2
  where Δchi2 uses df=2 parameters at conf (default 0.95).
  This is valid even when Nobs=2.

- GOF hypothesis test is ONLY defined when dof_gof = Nobs - Nparams >= 1.
  We store dof_gof, p_value, passes_gof but DO NOT use GOF to define acceptance-region IMFs.

Speed / caching:
- We generate isochrones ONLY for a 6-filter superset (JWST+HST) and reuse those
  interpolated tables for all runs by slicing/filtering in the chi2 calculation.

Usage:
  python run_all_fits.py \
      --dat /scratch/wyz5rge/synthetic-hr/data/phot_joined_avmass.dat \
      --out_root results_stepwise \
      --log_age 6.0 --metallicity 0 --dist 4500
"""

import os
import csv
import math
import time
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
    filt_list: List[str]          # informational; superset is used internally
    filters: List[str]            # isochrone column names used for mags-only
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
# Utility
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Plotting: identity/error
# -----------------------------
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

    # GOF visualization only; acceptance region is NOT based on GOF
    df_use["passes_gof_bool"] = df_use["passes_gof"] == True  # noqa: E712

    s = 6
    alpha = 0.35
    acc = df_use[df_use["passes_gof_bool"]]
    noacc = df_use[~df_use["passes_gof_bool"]]

    # Mass identity
    plt.figure(figsize=(7, 7))
    plt.scatter(noacc["true_mass"], noacc["best_mass"], s=s, alpha=alpha, label="fails GOF / undefined", marker="o")
    plt.scatter(acc["true_mass"], acc["best_mass"], s=s, alpha=alpha, label="passes GOF", marker="o")
    plt.plot([mass_xlim[0], mass_xlim[1]], [mass_xlim[0], mass_xlim[1]], "k--", lw=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(mass_xlim); plt.ylim(mass_xlim)
    plt.xlabel("Andersen (truth) Mass (M$_\\odot$)")
    plt.ylabel(f"{run_label} Mass (M$_\\odot$)")
    plt.title(f"Mass identity: {run_label} vs Andersen truth")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mass_identity.png"), dpi=200)
    plt.close()

    # Mass error hist
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(-200, 201, 10)
    plt.hist(df_use["mass_pct_error"].dropna(), bins=bins, alpha=0.8)
    plt.xlim(-200, 200)
    plt.xlabel("Mass % Error = 100*(fit - truth)/truth")
    plt.ylabel("Count")
    plt.title(f"Mass % Error histogram: {run_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mass_error_hist.png"), dpi=200)
    plt.close()

    # AV identity
    plt.figure(figsize=(7, 7))
    plt.scatter(noacc["true_AV"], noacc["best_AV"], s=s, alpha=alpha, label="fails GOF / undefined", marker="o")
    plt.scatter(acc["true_AV"], acc["best_AV"], s=s, alpha=alpha, label="passes GOF", marker="o")
    plt.plot([av_xlim[0], av_xlim[1]], [av_xlim[0], av_xlim[1]], "k--", lw=1)
    plt.xlim(av_xlim); plt.ylim(av_xlim)
    plt.xlabel("Andersen (truth) AV")
    plt.ylabel(f"{run_label} AV")
    plt.title(f"AV identity: {run_label} vs Andersen truth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "av_identity.png"), dpi=200)
    plt.close()

    # AV error hist
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(-200, 201, 10)
    plt.hist(df_use["av_pct_error"].dropna(), bins=bins, alpha=0.8)
    plt.xlim(-200, 200)
    plt.xlabel("AV % Error = 100*(fit - truth)/truth")
    plt.ylabel("Count")
    plt.title(f"AV % Error histogram: {run_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "av_error_hist.png"), dpi=200)
    plt.close()


# -----------------------------
# IMF helpers (matched-sample, 4 IMFs)
# -----------------------------
def _sanitize_masses(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return m[np.isfinite(m) & (m > 0)]


def make_log_bins_from_samples(samples: List[np.ndarray], nbins: int) -> np.ndarray:
    allm = np.concatenate([_sanitize_masses(s) for s in samples if s is not None and len(s) > 0], axis=0)
    allm = _sanitize_masses(allm)
    if allm.size == 0:
        return np.array([])
    m_min = max(0.01, float(np.min(allm)))
    m_max = float(np.max(allm))
    if not (np.isfinite(m_min) and np.isfinite(m_max)) or m_max <= m_min:
        return np.array([])
    return np.logspace(np.log10(m_min), np.log10(m_max), nbins + 1)


def imf_hist_on_edges(masses: np.ndarray, edges: np.ndarray):
    masses = _sanitize_masses(masses)
    if masses.size == 0 or edges.size < 2:
        return None
    N, _ = np.histogram(masses, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    dndm = N / widths
    return centers, N, dndm


def make_dndlogM_fixed_edges(masses: np.ndarray, edges_log: np.ndarray):
    masses = _sanitize_masses(masses)
    if masses.size == 0 or edges_log.size < 2:
        return None
    logM = np.log10(masses)
    N, _ = np.histogram(logM, bins=edges_log)
    centers_log = 0.5 * (edges_log[:-1] + edges_log[1:])
    centers_M = 10 ** centers_log
    binw = float(edges_log[1] - edges_log[0])
    dndlog = N / binw
    return centers_M, N, dndlog


def fit_powerlaw_slope_highmass(centers: np.ndarray, dndm: np.ndarray, m_break: float = 0.5):
    # Fit only above break mass
    mask = (centers >= m_break) & (dndm > 0) & np.isfinite(dndm) & np.isfinite(centers)
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


def fit_lognormal_lowmass(centers_M: np.ndarray, dndlog: np.ndarray, m_break: float = 0.5):
    from scipy.optimize import curve_fit
    mask = (centers_M <= m_break) & (dndlog > 0) & np.isfinite(dndlog) & np.isfinite(centers_M) & (centers_M > 0)
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


def plot_imfs_matched(
    df: pd.DataFrame,
    out_dir: str,
    run_label: str,
    nbins: int = 25,
    bin_width_dex: float = 0.10,
    m_break: float = 0.5,
) -> None:
    """
    Produces matched-sample IMF plots.

    Type 1 (ALL): stars present in df (i.e., fit produced output)
      - Andersen_all: true_mass on those same indices
      - Fit_all:      best_mass on those same indices

    Type 2 (PASS χ² GOF): subset where χ² GOF passes (passes_gof == True)
      - defined only when dof_gof >= 1 (i.e., Nobs - Nparams >= 1)
      - Andersen_pass: true_mass on those same indices
      - Fit_pass:      best_mass on those same indices

    New in this version:
      - Fit/overlay lognormal curves for PASS samples too (on dN/dlogM plot)
      - Overlay low-mass lognormal shapes on Counts and dN/dM plots (below m_break)
        using conversions from dN/dlog10M
    """
    ensure_dir(out_dir)

    d = df.copy()

    # Base filter: must have fit outputs and truth
    d = d[np.isfinite(d["best_mass"]) & (d["best_mass"] > 0)]
    d = d[np.isfinite(d["true_mass"]) & (d["true_mass"] > 0)]

    if len(d) == 0:
        return

    # Type 2 mask: robust to CSV round-tripping (bool/str/NaN)
    pass_mask = d["passes_gof"].astype(str).str.lower().isin(["true", "1"])
    d_all = d
    d_pass = d[pass_mask].copy()

    andersen_all = d_all["true_mass"].to_numpy()
    fit_all = d_all["best_mass"].to_numpy()
    andersen_pass = d_pass["true_mass"].to_numpy()
    fit_pass = d_pass["best_mass"].to_numpy()

    # Shared linear-mass bin edges for counts/dN/dM (use both ALL + PASS samples)
    edges = make_log_bins_from_samples([andersen_all, fit_all, andersen_pass, fit_pass], nbins=nbins)
    if edges.size < 2:
        return

    # Shared logM edges for dN/dlogM (use ALL only for axis range consistency)
    all_for_log = np.concatenate([_sanitize_masses(andersen_all), _sanitize_masses(fit_all)], axis=0)
    if all_for_log.size == 0:
        return
    log_lo = np.floor(np.log10(all_for_log.min()) / bin_width_dex) * bin_width_dex
    log_hi = np.ceil(np.log10(all_for_log.max()) / bin_width_dex) * bin_width_dex
    edges_log = np.arange(log_lo, log_hi + bin_width_dex, bin_width_dex)

    # Histograms on shared edges
    h_A_all = imf_hist_on_edges(andersen_all, edges)
    h_F_all = imf_hist_on_edges(fit_all, edges)
    h_A_pass = imf_hist_on_edges(andersen_pass, edges) if andersen_pass.size else None
    h_F_pass = imf_hist_on_edges(fit_pass, edges) if fit_pass.size else None

    if not h_A_all or not h_F_all:
        return

    # ---------
    # Helpers to overlay low-mass lognormal shapes on counts and dN/dM plots
    # ---------
    def _lognormal_counts_per_bin(edges_lin: np.ndarray, A: float, logmc: float, sig: float):
        centers = np.sqrt(edges_lin[:-1] * edges_lin[1:])               # geometric centers
        dlog = np.log10(edges_lin[1:]) - np.log10(edges_lin[:-1])       # bin widths in dex
        dndlog = lognormal_dndlogM(centers, A, logmc, sig)              # dN/dlog10M at centers
        Npred = dndlog * dlog                                           # predicted counts/bin
        return centers, Npred

    def _lognormal_dndm(M: np.ndarray, A: float, logmc: float, sig: float):
        # dN/dM = (1 / (M ln 10)) * dN/dlog10M
        return lognormal_dndlogM(M, A, logmc, sig) / (M * np.log(10.0))

    def overlay_ln_counts(fit, edges_lin: np.ndarray, label: str, linestyle: str = "-"):
        if not fit:
            return
        (A, logmc, sig), _ = fit
        centers, Npred = _lognormal_counts_per_bin(edges_lin, A, logmc, sig)
        mask = (centers <= m_break) & np.isfinite(Npred) & (Npred > 0)
        if np.sum(mask) < 2:
            return
        plt.plot(centers[mask], Npred[mask], linewidth=2, alpha=0.85, linestyle=linestyle, label=label)

    def overlay_ln_dndm(fit, label: str, linestyle: str = "-"):
        if not fit:
            return
        (A, logmc, sig), _ = fit
        # plot only up to break
        mmin = max(0.01, float(edges[0]))
        mmax = float(m_break)
        if not (np.isfinite(mmin) and np.isfinite(mmax)) or mmax <= mmin:
            return
        M = np.logspace(np.log10(mmin), np.log10(mmax), 300)
        y = _lognormal_dndm(M, A, logmc, sig)
        mask = np.isfinite(y) & (y > 0)
        if np.sum(mask) < 5:
            return
        plt.plot(M[mask], y[mask], linewidth=2, alpha=0.85, linestyle=linestyle, label=label)

    # ---------
    # Compute lognormal fits for ALL now (we'll also compute PASS later when available)
    # Use dN/dlogM binnings because that’s what fit_lognormal_lowmass expects.
    # ---------
    bA_all = make_dndlogM_fixed_edges(andersen_all, edges_log)
    bF_all = make_dndlogM_fixed_edges(fit_all, edges_log)
    lnA = None
    lnF = None
    if bA_all and bF_all:
        cMA, _, dAlog = bA_all
        cMF, _, dFlog = bF_all
        lnA = fit_lognormal_lowmass(cMA, dAlog, m_break=m_break)
        lnF = fit_lognormal_lowmass(cMF, dFlog, m_break=m_break)

    # Also compute PASS lognormal fits (for curve overlays) if PASS sample exists
    lnA_pass = None
    lnF_pass = None
    bA_pass = None
    bF_pass = None
    if andersen_pass.size and fit_pass.size:
        bA_pass = make_dndlogM_fixed_edges(andersen_pass, edges_log)
        bF_pass = make_dndlogM_fixed_edges(fit_pass, edges_log)
        if bA_pass and bF_pass:
            cMAp, _, dAlogp = bA_pass
            cMFp, _, dFlogp = bF_pass
            lnA_pass = fit_lognormal_lowmass(cMAp, dAlogp, m_break=m_break)
            lnF_pass = fit_lognormal_lowmass(cMFp, dFlogp, m_break=m_break)

    # =================
    # Counts plot
    # =================
    plt.figure(figsize=(10, 5))
    cA, NA, _ = h_A_all
    cF, NF, _ = h_F_all
    plt.step(cA, NA, where="mid", linewidth=2, label="Andersen (truth) — ALL (matched)")
    plt.step(cF, NF, where="mid", linewidth=2, linestyle="--", label=f"{run_label} — ALL")

    if h_A_pass and h_F_pass:
        cAp, NAp, _ = h_A_pass
        cFp, NFp, _ = h_F_pass
        plt.step(cAp, NAp, where="mid", linewidth=2, alpha=0.85, label="Andersen (truth) — PASS χ² GOF (matched)")
        plt.step(cFp, NFp, where="mid", linewidth=2, linestyle="--", alpha=0.85, label=f"{run_label} — PASS χ² GOF")
    else:
        # Distinguish "GOF undefined" vs "none passed"
        has_any_gof = np.any(d["dof_gof"].fillna(0).astype(int) >= 1)
        msg = "PASS sample empty (none passed χ² GOF)" if has_any_gof else "PASS sample undefined (dof_gof<1 for all)"
        plt.text(0.02, 0.95, msg, transform=plt.gca().transAxes, va="top")

    # Low-mass overlay fits (counts) — ALL and PASS
    if lnA:
        (A, logmc, sig), _ = lnA
        overlay_ln_counts(lnA, edges, f"Andersen LN counts (M≤{m_break:.2f})", linestyle="-")
    if lnF:
        (A, logmc, sig), _ = lnF
        overlay_ln_counts(lnF, edges, f"Fit LN counts (M≤{m_break:.2f})", linestyle="--")
    if lnA_pass:
        overlay_ln_counts(lnA_pass, edges, f"Andersen LN counts PASS (M≤{m_break:.2f})", linestyle=":")
    if lnF_pass:
        overlay_ln_counts(lnF_pass, edges, f"Fit LN counts PASS (M≤{m_break:.2f})", linestyle="-.")

    plt.axvline(m_break, linestyle=":", linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("Counts per bin")
    plt.title(f"IMF Counts (matched samples): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_counts_matched.png"), dpi=200)
    plt.close()

    # =================
    # dN/dM plot + powerlaw fits above break + lognormal below break
    # =================
    plt.figure(figsize=(10, 6))
    cA, _, dA = h_A_all
    cF, _, dF = h_F_all
    plt.step(cA, dA, where="mid", linewidth=2, label="Andersen — ALL")
    plt.step(cF, dF, where="mid", linewidth=2, linestyle="--", label=f"{run_label} — ALL")

    fitA = fit_powerlaw_slope_highmass(cA, dA, m_break=m_break)
    fitF = fit_powerlaw_slope_highmass(cF, dF, m_break=m_break)

    def overlay_pl(fit, centers, label):
        if not fit:
            return
        alpha, slope, intercept = fit
        mmin = max(m_break, float(np.nanmin(centers)))
        mmax = float(np.nanmax(centers))
        if not (np.isfinite(mmin) and np.isfinite(mmax)) or mmax <= mmin:
            return
        M = np.logspace(np.log10(mmin), np.log10(mmax), 300)
        y = 10 ** (intercept + slope * np.log10(M))
        plt.plot(M, y, linewidth=2, alpha=0.8, label=label)

    if fitA:
        overlay_pl(fitA, cA, f"Andersen PL (M≥{m_break:.2f}): α={fitA[0]:.2f}")
    if fitF:
        overlay_pl(fitF, cF, f"Fit PL (M≥{m_break:.2f}): α={fitF[0]:.2f}")

    if h_A_pass and h_F_pass:
        cAp, _, dAp = h_A_pass
        cFp, _, dFp = h_F_pass
        plt.step(cAp, dAp, where="mid", linewidth=2, alpha=0.85, label="Andersen — PASS χ² GOF")
        plt.step(cFp, dFp, where="mid", linewidth=2, linestyle="--", alpha=0.85, label=f"{run_label} — PASS χ² GOF")

    # Low-mass overlay fits (dN/dM) — ALL and PASS
    if lnA:
        (A, logmc, sig), _ = lnA
        overlay_ln_dndm(lnA, f"Andersen LN dN/dM (M≤{m_break:.2f})", linestyle="-")
    if lnF:
        (A, logmc, sig), _ = lnF
        overlay_ln_dndm(lnF, f"Fit LN dN/dM (M≤{m_break:.2f})", linestyle="--")
    if lnA_pass:
        overlay_ln_dndm(lnA_pass, f"Andersen LN dN/dM PASS (M≤{m_break:.2f})", linestyle=":")
    if lnF_pass:
        overlay_ln_dndm(lnF_pass, f"Fit LN dN/dM PASS (M≤{m_break:.2f})", linestyle="-.")

    plt.axvline(m_break, linestyle=":", linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("dN/dM")
    plt.title(f"IMF dN/dM (matched samples): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_dndm_matched.png"), dpi=200)
    plt.close()

    # =================
    # dN/dlogM plot + lognormal fits below break (ALL + PASS curves)
    # =================
    # (Re-use the bA_all/bF_all already computed above)
    if (bA_all is None) or (bF_all is None):
        return

    cMA, _, dAlog = bA_all
    cMF, _, dFlog = bF_all

    plt.figure(figsize=(10, 7))
    plt.scatter(cMA, dAlog, s=25, alpha=0.75, label="Andersen — ALL")
    plt.scatter(cMF, dFlog, s=25, alpha=0.75, label=f"{run_label} — ALL")

    M_plot = np.logspace(np.log10(float(np.min(cMA))), np.log10(float(np.max(cMA))), 500)

    def overlay_ln(fit, label, linestyle: str = "-"):
        if not fit:
            return
        (A, logmc, sig), _ = fit
        plt.plot(
            M_plot,
            lognormal_dndlogM(M_plot, A, logmc, sig),
            linewidth=2,
            alpha=0.85,
            linestyle=linestyle,
            label=label,
        )

    if lnA:
        (A, logmc, sig), _ = lnA
        overlay_ln(lnA, f"Andersen LN (M≤{m_break:.2f}): mc={10**logmc:.2f}, σ={sig:.2f}", linestyle="-")
    if lnF:
        (A, logmc, sig), _ = lnF
        overlay_ln(lnF, f"Fit LN (M≤{m_break:.2f}): mc={10**logmc:.2f}, σ={sig:.2f}", linestyle="--")

    if andersen_pass.size and fit_pass.size and bA_pass and bF_pass:
        cMAp, _, dAlogp = bA_pass
        cMFp, _, dFlogp = bF_pass
        plt.scatter(cMAp, dAlogp, s=25, alpha=0.75, label="Andersen — PASS χ² GOF")
        plt.scatter(cMFp, dFlogp, s=25, alpha=0.75, label=f"{run_label} — PASS χ² GOF")

        # NEW: overlay lognormal curves for PASS samples too
        if lnA_pass:
            (A, logmc, sig), _ = lnA_pass
            overlay_ln(lnA_pass, f"Andersen LN PASS (M≤{m_break:.2f}): mc={10**logmc:.2f}, σ={sig:.2f}", linestyle=":")
        if lnF_pass:
            (A, logmc, sig), _ = lnF_pass
            overlay_ln(lnF_pass, f"Fit LN PASS (M≤{m_break:.2f}): mc={10**logmc:.2f}, σ={sig:.2f}", linestyle="-.")

    plt.axvline(m_break, linestyle=":", linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("dN/dlog$_{10}$M")
    plt.title(f"IMF dN/dlogM (matched samples): {run_label}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imf_dndlogM_lognormal_matched.png"), dpi=200)
    plt.close()

    # =================
    # Summary text
    # =================
    summary_path = os.path.join(out_dir, "imf_matched_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run: {run_label}\n")
        f.write(f"ALL matched N:        {len(d_all)}\n")
        f.write(f"PASS χ² GOF matched N: {len(d_pass)}\n\n")

        if fitA:
            f.write(f"Andersen powerlaw (M≥{m_break}): alpha={fitA[0]:.6f}\n")
        else:
            f.write("Andersen powerlaw: insufficient bins above break\n")
        if fitF:
            f.write(f"Fit powerlaw (M≥{m_break}): alpha={fitF[0]:.6f}\n")
        else:
            f.write("Fit powerlaw: insufficient bins above break\n")

        f.write("\n")
        if lnA:
            (A, logmc, sig), _ = lnA
            f.write(f"Andersen lognormal (M≤{m_break}): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Andersen lognormal: insufficient bins below break\n")
        if lnF:
            (A, logmc, sig), _ = lnF
            f.write(f"Fit lognormal (M≤{m_break}): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Fit lognormal: insufficient bins below break\n")

        f.write("\n")
        if lnA_pass:
            (A, logmc, sig), _ = lnA_pass
            f.write(f"Andersen lognormal PASS (M≤{m_break}): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Andersen lognormal PASS: insufficient bins below break or PASS empty\n")
        if lnF_pass:
            (A, logmc, sig), _ = lnF_pass
            f.write(f"Fit lognormal PASS (M≤{m_break}): mc={10**logmc:.6f} Msun, sigma={sig:.6f} dex\n")
        else:
            f.write("Fit lognormal PASS: insufficient bins below break or PASS empty\n")
            

# -----------------------------
# Per-star AV-mass diagnostic plot
# -----------------------------
def plot_av_mass_acceptance(
    arr: np.ndarray,
    acceptable: np.ndarray,
    best: np.void,
    out_path: str,
    title: str,
    top_n: int = 300,
    true_av: Optional[float] = None,
    true_mass: Optional[float] = None,
) -> None:
    """
    arr: dtype [("AV","AKs","mass","chi2")]
    acceptable: subset of arr within threshold (may be empty)
    best: one row
    If acceptable empty, show the lowest top_n chi2 points.
    Optionally overlays Andersen truth (true_mass, true_av).
    """
    ensure_dir(os.path.dirname(out_path))

    a = arr[np.isfinite(arr["chi2"]) & np.isfinite(arr["AV"]) & np.isfinite(arr["mass"])]
    if a.size == 0:
        return

    # pick points to display
    if acceptable is not None and acceptable.size > 0:
        show = acceptable
        subtitle = f"accept pts={acceptable.size}"
    else:
        k = min(top_n, a.size)
        idx = np.argpartition(a["chi2"], kth=k - 1)[:k]
        show = a[idx]
        subtitle = f"no accept region; showing lowest {k} chi2 pts"

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(show["mass"], show["AV"], c=show["chi2"], s=10, alpha=0.75)
    plt.colorbar(sc, label=r"$\chi^2$")

    # Best fit
    plt.scatter(
        [best["mass"]], [best["AV"]],
        marker="*", s=220, edgecolor="k", linewidth=0.8,
        label="best fit",
        zorder=5,
    )

    # Andersen truth (overlay)
    if true_av is not None and true_mass is not None:
        if np.isfinite(true_av) and np.isfinite(true_mass) and (true_av >= 0) and (true_mass > 0):
            plt.scatter(
                [true_mass], [true_av],
                marker="X", s=140, edgecolor="k", linewidth=0.8,
                label="Andersen truth",
                zorder=6,
            )

    plt.xscale("log")
    plt.xlabel("Mass (M$_\\odot$)")
    plt.ylabel("AV")
    plt.title(f"{title}\n{subtitle}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Superset-grid fitter (reuse JWST+HST iso for all runs)
# -----------------------------
class SupersetGridFitter:
    """
    Generates a single (AV, mass) grid using a 6-filter superset isochrone,
    then evaluates chi2 for each run by slicing that same model grid.
    """

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
        lock_timeout_s: float = 600.0,
    ):
        self.iso_dir = iso_dir
        self.dist = dist
        self.metallicity = metallicity
        self.log_age = log_age
        self.interp_n = interp_n
        self.av_grid_halfwidth = av_grid_halfwidth
        self.av_grid_step = av_grid_step
        self.conf = conf
        self.lock_timeout_s = lock_timeout_s

        ensure_dir(self.iso_dir)

        self.evo_model = evolution.Baraffe15()
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawCardelli(3.1)

        # microns
        self.filter_wavelengths = {
            "m_jwst_F162M": 1.62,
            "m_jwst_F182M": 1.82,
            "m_jwst_F200W": 2.00,
            "m_hst_f125w":  1.25,
            "m_hst_f139m":  1.39,
            "m_hst_f160w":  1.60,
        }

        self.AV_to_AKs = 1.0 / 0.1179
        self.AKs_per_AV = 0.118

        self.delta_chi2 = float(chi2.ppf(self.conf, df=2))

        self._lock_path = os.path.join(self.iso_dir, ".spisea_cache.lock")

    # --- basic file lock to avoid partial-write races in SPISEA cache ---
    def _acquire_lock(self):
        import fcntl
        start = time.time()
        f = open(self._lock_path, "a+")
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                if (time.time() - start) > self.lock_timeout_s:
                    f.close()
                    raise TimeoutError(f"Timed out acquiring lock: {self._lock_path}")
                time.sleep(0.2)

    @staticmethod
    def _release_lock(f):
        import fcntl
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

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

    def minimize_AKs_from_162_182(self, m162: float, m182: float) -> float:
        """
        Initial guess for AV via minimizing CMD distance on unreddened superset isochrone.
        """
        lock = self._acquire_lock()
        try:
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
        finally:
            self._release_lock(lock)

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
        plots_root_by_run: Dict[str, str],
        plot_top_n: int = 300,
    ) -> Dict[str, Dict[str, object]]:
        """
        Compute results for only the runs in runs_needed for this star.
        Returns dict: run_name -> result row dict.
        Also writes a per-star AV-mass plot into each run's plots/ folder.
        """

        # AV guess inputs
        m162, m182 = row["F162M"], row["F182M"]
        aks_guess = self.minimize_AKs_from_162_182(m162, m182)
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

        # Run-specific chi2 lists
        results_by_run: Dict[str, List[Tuple[float, float, float, float]]] = {cfg.name: [] for cfg in runs_needed}

        # Grid search (superset model reused; note: still loops over AV, but avoids per-run iso generation)
        for av in av_grid:
            aks = av * self.AKs_per_AV

            lock = self._acquire_lock()
            try:
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
            finally:
                self._release_lock(lock)

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

            # GOF (only if dof>=1)
            _, _, nobs = obs_by_run[cfg.name]
            dof_gof = int(nobs - 2)
            passes_gof = None
            p_value = None
            if dof_gof >= 1:
                p_value = float(chi2.sf(min_chi2, df=dof_gof))
                passes_gof = bool(p_value > (1.0 - self.conf))

            # Bounds from acceptance region
            if acceptable.size == 0:
                mass_min = mass_max = av_min = av_max = None
                intersects = None
            else:
                mass_min = float(np.min(acceptable["mass"]))
                mass_max = float(np.max(acceptable["mass"]))
                av_min = float(np.min(acceptable["AV"]))
                av_max = float(np.max(acceptable["AV"]))
                intersects = (av_min <= row["true_AV"] <= av_max) and (mass_min <= row["true_mass"] <= mass_max)

            # Per-star diagnostic plot -> run/plots/
            plot_dir = plots_root_by_run[cfg.name]
            out_path = os.path.join(plot_dir, f"idx_{index_noncomment:06d}_lineno_{file_lineno:06d}.png")
            plot_av_mass_acceptance(
                arr=arr,
                acceptable=acceptable,
                best=best,
                out_path=out_path,
                title=f"{cfg.name} idx={index_noncomment} lineno={file_lineno}",
                top_n=plot_top_n,
                true_av=float(row["true_AV"]),
                true_mass=float(row["true_mass"]),
            )

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
# Orchestration I/O
# -----------------------------
def setup_run_io(out_root: str, runs: List[FitRunConfig]) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, logging.Logger],
    Dict[str, set],
]:
    """
    Returns:
      csv_path_by_run, plots_dir_by_run, logger_by_run, processed_index_set_by_run
    """
    csv_paths: Dict[str, str] = {}
    plots_dirs: Dict[str, str] = {}
    loggers: Dict[str, logging.Logger] = {}
    processed: Dict[str, set] = {}

    for cfg in runs:
        run_dir = os.path.join(out_root, cfg.name)
        ensure_dir(run_dir)
        ensure_dir(os.path.join(run_dir, "compare_identity"))
        ensure_dir(os.path.join(run_dir, "imf"))
        plots_dir = os.path.join(run_dir, "plots")
        ensure_dir(plots_dir)
        plots_dirs[cfg.name] = plots_dir

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

    return csv_paths, plots_dirs, loggers, processed


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


# -----------------------------
# Main
# -----------------------------
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
    parser.add_argument("--conf", type=float, default=0.95)
    parser.add_argument("--plot_top_n", type=int, default=300, help="If no accept region, plot lowest-N chi2 points")

    parser.add_argument("--nbins_logM", type=int, default=25)
    parser.add_argument("--bin_width_dex", type=float, default=0.10)
    parser.add_argument("--m_break", type=float, default=0.5)

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

    csv_paths, plots_dirs, loggers, processed_by_run = setup_run_io(args.out_root, runs)

    # Shared superset iso cache
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

    # mapping for mags-only runs
    filt_to_dat = {
        "m_jwst_F162M": ("F162M", "e162"),
        "m_jwst_F182M": ("F182M", "e182"),
        "m_jwst_F200W": ("F200W", "e200"),
        "m_hst_f125w":  ("F125W", "e125"),
        "m_hst_f139m":  ("F139M", "e139"),
        "m_hst_f160w":  ("F160W", "e160"),
    }

    # Pre-scan truth
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

    # Main pass
    idx_noncomment = -1
    with open(args.dat, "r") as f:
        for lineno_1based, line in enumerate(f, start=1):
            if line.startswith("#") or (not line.strip()):
                continue
            idx_noncomment += 1

            runs_needed = [cfg for cfg in runs if idx_noncomment not in processed_by_run[cfg.name]]
            if not runs_needed:
                continue

            parts = [float(x) for x in line.split()]
            if len(parts) < 16:
                for cfg in runs_needed:
                    loggers[cfg.name].info("Skip idx=%d lineno=%d (too few columns: %d)", idx_noncomment, lineno_1based, len(parts))
                continue

            row = parse_phot_line(parts)

            # per-run runner predicate
            runs_ready: List[FitRunConfig] = []
            for cfg in runs_needed:
                if cfg.mode == "cmd_a1":
                    mags = [row["F162M"], row["F182M"]]
                    errs = [row["e162"], row["e182"]]
                else:
                    mags, errs = [], []
                    for f_iso in cfg.filters:
                        m_key, e_key = filt_to_dat[f_iso]
                        mags.append(row[m_key])
                        errs.append(row[e_key])

                if should_skip_runner_style(mags, errs, row["true_AV"], row["true_mass"]):
                    loggers[cfg.name].info("Skip idx=%d lineno=%d (runner predicate)", idx_noncomment, lineno_1based)
                    processed_by_run[cfg.name].add(idx_noncomment)
                    continue

                runs_ready.append(cfg)

            if not runs_ready:
                continue

            try:
                results = fitter.analyze_star_for_runs(
                    index_noncomment=idx_noncomment,
                    file_lineno=lineno_1based,
                    row=row,
                    runs_needed=runs_ready,
                    filt_to_dat=filt_to_dat,
                    plots_root_by_run=plots_dirs,
                    plot_top_n=args.plot_top_n,
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

    # Postprocess per run
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

            plot_identity_and_error_hist(
                df=df,
                out_dir=os.path.join(run_dir, "compare_identity"),
                run_label=cfg.name,
                mass_xlim=mass_xlim,
                av_xlim=av_xlim,
            )

            # Matched-sample IMFs (Andersen vs Fit; all vs accept-only)
            plot_imfs_matched(
                df=df,
                out_dir=os.path.join(run_dir, "imf"),
                run_label=cfg.name,
                nbins=args.nbins_logM,
                bin_width_dex=args.bin_width_dex,
                m_break=args.m_break,
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