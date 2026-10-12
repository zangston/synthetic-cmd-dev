# %% [markdown]
# # Cluster mass and cloud surface density: seed 00, epsilon_ff = 0.03
#
# Jupytext percent-format script; run as a notebook or with Python.
#
# Simulations: M_cl = 300, 3000, 30000 Msun, each at Sigma_cloud = 0.1 and
# 1.0 g cm^-2, epsilon_ff = 0.03, seed 00.
#
# Paper figures:
#   (1) F182M-F200W vs F200W time evolution: 3 masses x 6 epochs, one figure
#       per Sigma_cloud, with shared axes and no overlapping legend.
#   (2) Four spread metrics (H-R + 3 CMDs) vs time, in 2x4 (both Sigmas)
#       and 2x2 (individual Sigma) figures.
#   (3) Formation histories and peak-spread diagnostic, to TEST rather than
#       assume whether mass prolongs star formation and increases spread.
#
# IMPORTANT: This is the original primary-star/one-row-per-current-system
# analysis from the supplied mass script, NOT a resolved/unresolved binary
# photometry analysis. Current systems come from converter.to_spicea_table().
# Binary companion flux is not combined. The H-R diagram remains physical.
#
# Metric: Teff in [3000,3600] K; six fixed 100 K H-R bins or six fixed
# equal-width CMD COLOR bins anchored to the 1 Myr isochrone endpoints;
# mean(upper quartile tail)-mean(lower quartile tail) in each bin; mean of
# available valid bins (>=8 stars each). Sparse-bin coverage is reported.
#
# This version fixes the old >2-filter interpolator bug. The custom
# interpolator accepts EXACTLY TWO photometric columns per call and returns
# (L, Teff, logg, mag1, mag2). We call it for three FILTER PAIRS and combine
# the results. The additional two calls are needed only for stars in the
# 3000-3600 K band, which saves substantial time for M_cl=30000 Msun.

# %%
from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # safe under jupyter-nbconvert / batch jobs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.table import Column, Table
import numpy as np
import pandas as pd

from spisea import atmospheres, reddening, synthetic
from nbody6tools import Reader
from nbody62spisea import converter

sys.path.insert(0, '/home/wyz5rge/synthetic-cmd-dev/cmd_generator')
import interpolator  # noqa: E402

# %% [markdown]
# ## Configuration

# %%
SIM_ROOT = Path('/standard/Tan_JC/backup_protoclusters/multiples')
MASS_DIRS = {300: 'M300', 3000: 'M3000new', 30000: 'M30000'}
SIGMA_DIRS = {0.1: 'sigma0p1', 1.0: 'sigma1p0'}
EPSILON_FF = 0.03
SFE_DIR = 'sfe_ff003'
SEED = '00'
SIMULATION_PATHS = {
    (sigma, mass): SIM_ROOT / mass_dir / sigma_dir / 'fiducial' / SFE_DIR / SEED
    for sigma, sigma_dir in SIGMA_DIRS.items()
    for mass, mass_dir in MASS_DIRS.items()
}

UPDATED_MERGED_ROOT = Path(
    '/home/wyz5rge/SPISEA/evolution/merged/baraffe_pisa_ekstrom_parsec/'
)
OUTPUT_DIR = Path.cwd() / 'cluster_mass_comparison_seed00_outputs'
CACHE_DIR = Path.cwd() / 'cluster_mass_comparison_seed00_cache'
ISO_CACHE_DIR = Path.cwd() / 'iso_cache'  # existing cache; NEVER deleted
RECOMPUTE_INTERPOLATION = False
SAVE_PNG = True
SAVE_PDF = True
SHOW_FIGURES = False  # batch-safe; set True for interactive notebooks

USE_ROTATING_MERGED = False
AKS = 0.0
DISTANCE_PC = 410.0
METALLICITY = 0.0
ATM_FUNC = atmospheres.get_BTSettl_2015_atmosphere
RED_LAW = reddening.RedLawHosek18b()
ISO_AGES_MYR = np.arange(1.0, 20.0 + 0.25, 0.5)
ISO_LOG_AGES = np.log10(ISO_AGES_MYR * 1e6)
ANALYSIS_TIMES_MYR = np.arange(1.0, 20.0 + 0.25, 0.5)
# Columns in the paper's mass x epoch CMD figures:
DISPLAY_TIMES_MYR = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
SNAPSHOT_TIME_TOLERANCE_MYR = 0.26  # avoid silently reusing a distant snapshot

TEFF_MIN_K = 3000.0
TEFF_MAX_K = 3600.0
N_BINS = 6
MIN_STARS_PER_BIN = 8
REFERENCE_BIN_AGE_MYR = 1.0
CLIP_YOUNG_TO_GRID = True  # keep original treatment; report clipped fraction
L_SUN_WATTS = 3.846e26

FILTER_OBSMODES = {
    'F070W': 'jwst,F070W',
    'F182M': 'jwst,F182M',
    'F200W': 'jwst,F200W',
    'F555W': 'wfc3,uvis1,f555w',
    'F814W': 'wfc3,uvis1,f814w',
}
# The FIRST pair supplies the entire F182M-F200W CMD and the physical H-R
# coordinates for every star. Remaining pairs are only needed in the Teff band.
FILTER_PAIR_NAMES = (
    ('F182M', 'F200W'),
    ('F070W', 'F200W'),
    ('F555W', 'F814W'),
)
MASS_COLORS = {300: '#3467A7', 3000: '#D17B28', 30000: '#328468'}

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

for folder in (OUTPUT_DIR, CACHE_DIR, ISO_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
missing_paths = [(key, path) for key, path in SIMULATION_PATHS.items() if not path.is_dir()]
if missing_paths:
    msg = '\n'.join(f'  Sigma={s}, M={m}: {p}' for (s, m), p in missing_paths)
    raise FileNotFoundError('Missing simulation directories (check paths):\n' + msg)
print('Simulation inputs:')
for (sigma, mass), path in SIMULATION_PATHS.items():
    print(f'  Sigma={sigma:g}, M_cl={mass:5d}: {path}')
print(f'Isochrone cache (reused if compatible): {ISO_CACHE_DIR}')


@dataclass(frozen=True)
class Diagram:
    key: str
    title: str
    xlabel: str
    ylabel: str
    kind: str
    blue: str | None = None
    red: str | None = None
    y_filter: str | None = None


DIAGRAMS = [
    Diagram('hr', r'H-R: $\log_{10}(L/L_\odot)$ vs. $T_{\rm eff}$',
            r'$T_{\rm eff}$ [K]', r'$\log_{10}(L/L_\odot)$', 'hr'),
    Diagram('f070w_f200w', 'JWST F070W - F200W',
            'F070W - F200W', 'F200W', 'cmd', 'F070W', 'F200W', 'F200W'),
    Diagram('f182m_f200w', 'JWST F182M - F200W',
            'F182M - F200W', 'F200W', 'cmd', 'F182M', 'F200W', 'F200W'),
    Diagram('hst_f555w_f814w', 'HST F555W - F814W',
            'F555W - F814W', 'F814W', 'cmd', 'F555W', 'F814W', 'F814W'),
]
DIAGRAM_BY_KEY = {d.key: d for d in DIAGRAMS}


def save_figure(fig, stem):
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def normalized(s):
    return ''.join(c.lower() for c in str(s) if c.isalnum())


# %% [markdown]
# ## Merged evolutionary model and photometric isochrones

# %%
class MergedBaraffePisaEkstromParsecDAT:
    """Same .dat model reader as the supplied mass-comparison script."""

    def __init__(self, root_dir: Path | str, rot=False):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.rot = bool(rot)
        self.model_dir = str(self.root_dir)
        self.z_list = [0.015]
        self.z_solar = 0.015
        self.mass_list = []
        self.grid_dir = self.root_dir / ('z015_rot' if rot else 'z015_norot')
        if not self.grid_dir.is_dir():
            raise FileNotFoundError(self.grid_dir)
        self.age_file_map = {}
        for p in sorted(self.grid_dir.glob('iso_*.dat')):
            try:
                age = float(p.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            self.age_file_map[round(age, 2)] = p
        if not self.age_file_map:
            raise FileNotFoundError(f'No iso_*.dat in {self.grid_dir}')
        self.age_list = np.array(sorted(self.age_file_map), float)

    def isochrone(self, age=1e6, metallicity=0.0):
        requested = math.log10(age)
        if requested < self.age_list[0] or requested > self.age_list[-1]:
            raise ValueError(f'logAge {requested:.4f} outside grid')
        selected = float(self.age_list[np.argmin(np.abs(self.age_list - requested))])
        path = self.age_file_map[round(selected, 2)]
        dtype = [
            ('mass', 'f8'), ('logT', 'f8'), ('logL', 'f8'), ('logg', 'f8'),
            ('logT_WR', 'f8'), ('mass_current', 'f8'), ('phase', 'i4'),
            ('model_ref', 'U32'),
        ]
        data = np.genfromtxt(path, comments='#', dtype=dtype, encoding='utf-8')
        iso = Table(np.atleast_1d(data))
        iso.add_column(Column(
            ~np.isclose(np.asarray(iso['logT'], float),
                        np.asarray(iso['logT_WR'], float), rtol=0, atol=1e-8),
            name='isWR',
        ))
        iso.meta.update({
            'log_age': selected, 'log_age_requested': requested,
            'metallicity_in': metallicity, 'metallicity_act': 0.0,
            'source_file': str(path),
        })
        return iso


@dataclass
class IsoGrid:
    ages_myr: np.ndarray
    log_ages: np.ndarray
    isochrones: list
    coverage: pd.DataFrame
    filter_columns: dict[str, str]


def resolve_filter_column(colnames, filter_name):
    target = normalized(filter_name)
    candidates = [
        col for col in colnames
        if normalized(col).startswith('m') and normalized(col).endswith(target)
    ]
    instrument = 'hst' if filter_name in ('F555W', 'F814W') else 'jwst'
    preferred = [c for c in candidates if instrument in normalized(c)]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f'Cannot resolve {filter_name}: {candidates}')


def build_iso_grid(evo_model):
    isos, records, resolved = [], [], None
    for age, log_age in zip(ISO_AGES_MYR, ISO_LOG_AGES):
        print(f'Isochrone: {age:4.1f} Myr (load cached or generate if needed)', flush=True)
        try:
            iso = synthetic.IsochronePhot(
                log_age, AKS, DISTANCE_PC, metallicity=METALLICITY,
                evo_model=evo_model, atm_func=ATM_FUNC, red_law=RED_LAW,
                filters=list(FILTER_OBSMODES.values()), iso_dir=str(ISO_CACHE_DIR),
            )
            cols = {
                name: resolve_filter_column(iso.points.colnames, name)
                for name in FILTER_OBSMODES
            }
            if resolved is None:
                resolved = cols
                print('Magnitude columns:', resolved)
            elif resolved != cols:
                raise RuntimeError(f'Inconsistent filter columns: {cols} vs {resolved}')
            mass = np.asarray(iso.points['mass'], float)
            if len(mass) < 2 or not np.all(np.isfinite(mass)):
                raise RuntimeError('Invalid or empty isochrone mass grid')
            isos.append(iso)
            records.append(dict(age_myr=age, status='success',
                                n_points=len(mass), mass_min=float(mass.min()),
                                mass_max=float(mass.max()), error=''))
        except Exception as exc:
            isos.append(None)
            records.append(dict(age_myr=age, status='failed', n_points=0,
                                mass_min=np.nan, mass_max=np.nan, error=repr(exc)))
            print(f'  FAILED: {type(exc).__name__}: {exc}', flush=True)
    coverage = pd.DataFrame(records)
    coverage.to_csv(OUTPUT_DIR / 'isochrone_coverage.csv', index=False)
    failed = coverage[coverage.status != 'success']
    if len(failed):
        raise RuntimeError(
            'Isochrone failures. See isochrone_coverage.csv. The presence of '
            'iso_cache alone does not guarantee matching age/filter/model '
            'entries. Existing cache was NOT deleted.\n'
            + failed.to_string(index=False)
        )
    return IsoGrid(ISO_AGES_MYR.copy(), ISO_LOG_AGES.copy(), isos,
                   coverage, resolved)


evo_model = MergedBaraffePisaEkstromParsecDAT(
    UPDATED_MERGED_ROOT, rot=USE_ROTATING_MERGED
)
ISO_GRID = build_iso_grid(evo_model)
FILTER_PAIR_KEYS = tuple(
    (ISO_GRID.filter_columns[a], ISO_GRID.filter_columns[b])
    for a, b in FILTER_PAIR_NAMES
)


# %% [markdown]
# ## Fixed bin geometry: identical for all six simulation cases

# %%
def isochrone_xy(iso, diagram):
    teff = np.asarray(iso.points['Teff'], float)
    if diagram.kind == 'hr':
        lum = np.asarray(iso.points['L'], float)
        y = np.full_like(lum, np.nan)
        good = np.isfinite(lum) & (lum > 0)
        y[good] = np.log10(lum[good] / L_SUN_WATTS)
        return teff, y, teff
    a = np.asarray(iso.points[ISO_GRID.filter_columns[diagram.blue]], float)
    b = np.asarray(iso.points[ISO_GRID.filter_columns[diagram.red]], float)
    y = np.asarray(iso.points[ISO_GRID.filter_columns[diagram.y_filter]], float)
    return a - b, y, teff


def color_at_teff(iso, diagram, target, half_width=250.0):
    color, _, teff = isochrone_xy(iso, diagram)
    mask = (np.isfinite(color) & np.isfinite(teff)
            & (teff >= target - half_width)
            & (teff <= target + half_width))
    t, c = teff[mask], color[mask]
    if len(t) < 2:
        raise RuntimeError(f'Insufficient isochrone points near {target} K: {diagram.key}')
    order = np.argsort(t)
    t, c = t[order], c[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    sums = np.bincount(inverse, weights=c)
    counts = np.bincount(inverse)
    unique_c = sums / counts
    if not (unique_t[0] <= target <= unique_t[-1]):
        raise RuntimeError(f'Teff {target} K not bracketed: {diagram.key}')
    return float(np.interp(target, unique_t, unique_c))


STATIC_BINS = {}
for diagram in DIAGRAMS:
    if diagram.kind == 'hr':
        edges = np.linspace(TEFF_MIN_K, TEFF_MAX_K, N_BINS + 1)
        x3000, x3600 = TEFF_MIN_K, TEFF_MAX_K
    else:
        ref = ISO_GRID.isochrones[0]
        x3000 = color_at_teff(ref, diagram, TEFF_MIN_K)
        x3600 = color_at_teff(ref, diagram, TEFF_MAX_K)
        edges = np.linspace(min(x3000, x3600), max(x3000, x3600), N_BINS + 1)
    STATIC_BINS[diagram.key] = edges
    print(f'Fixed bins {diagram.key}: {np.round(edges, 5)}')

pd.DataFrame([
    dict(diagram=d.key, edge_index=i, x_edge=float(v),
         reference_age_myr=REFERENCE_BIN_AGE_MYR,
         sigma_independent=True)
    for d in DIAGRAMS for i, v in enumerate(STATIC_BINS[d.key])
]).to_csv(OUTPUT_DIR / 'fixed_bin_edges.csv', index=False)


# %% [markdown]
# ## Snapshot loading, pairwise interpolation, and resumable catalog cache

# %%
CACHE_CONFIG = dict(
    cache_format='pairwise_three_pairs_band_optimized_primary_only_2026_10',
    sim_paths={f'{s}:{m}': str(p) for (s, m), p in SIMULATION_PATHS.items()},
    ages=ISO_AGES_MYR.tolist(), distance_pc=DISTANCE_PC, aks=AKS,
    metallicity=METALLICITY, rotating=USE_ROTATING_MERGED,
    evo_root=str(UPDATED_MERGED_ROOT),
    filters=FILTER_OBSMODES, pairs=FILTER_PAIR_NAMES,
    clip_young=CLIP_YOUNG_TO_GRID,
    teff_selection=[TEFF_MIN_K, TEFF_MAX_K],
)
CACHE_SIGNATURE = hashlib.sha256(
    json.dumps(CACHE_CONFIG, sort_keys=True).encode('utf-8')
).hexdigest()[:16]
print('Interpolation cache signature:', CACHE_SIGNATURE)

CATALOG_COLUMNS = [
    'mass', 'age_myr', 'age_used_myr', 'age_clipped_to_grid',
    'teff', 'luminosity_watts', 'log_luminosity_lsun', 'logg',
    *[f'mag_{name}' for name in FILTER_OBSMODES],
]


def safe_interpolate_pair(age_myr, mass, filter_pair):
    if len(filter_pair) != 2:
        raise ValueError('Custom interpolator requires EXACTLY two filter columns')
    try:
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter('ignore')
            result = interpolator.interpolate(
                float(age_myr), float(mass), ISO_GRID.isochrones,
                ISO_GRID.log_ages, list(filter_pair),
            )
        if result is None:
            return None
        result = np.asarray(result, float).ravel()
        return result if result.size == 5 and np.all(np.isfinite(result)) else None
    except Exception:
        return None


def interpolate_population(masses, ages):
    """All stars: physical + F182M/F200W. Band stars: other 3 filters."""
    counters = dict(n_input=len(masses), nonfinite_mass=0, nonpositive_mass=0,
                    nonfinite_age=0, younger_than_grid=0, older_than_grid=0,
                    n_clipped_young=0, primary_pair_failed=0,
                    nonpositive_luminosity=0, n_in_temperature_band=0,
                    auxiliary_pair_failed=0, retained=0)
    rows = []
    amin, amax = ISO_GRID.ages_myr[0], ISO_GRID.ages_myr[-1]
    for mass, age in zip(masses, ages):
        mass, age = float(mass), float(age)
        if not np.isfinite(mass):
            counters['nonfinite_mass'] += 1
            continue
        if mass <= 0:
            counters['nonpositive_mass'] += 1
            continue
        if not np.isfinite(age):
            counters['nonfinite_age'] += 1
            continue
        used_age = age
        if used_age < amin:
            counters['younger_than_grid'] += 1
            if not CLIP_YOUNG_TO_GRID:
                continue
            used_age = float(amin)
            counters['n_clipped_young'] += 1
        if used_age > amax:
            counters['older_than_grid'] += 1
            continue
        base = safe_interpolate_pair(used_age, mass, FILTER_PAIR_KEYS[0])
        if base is None:
            counters['primary_pair_failed'] += 1
            continue
        luminosity, teff, logg, m182, m200 = map(float, base)
        if luminosity <= 0:
            counters['nonpositive_luminosity'] += 1
            continue
        row = dict(mass=mass, age_myr=age, age_used_myr=used_age,
                   age_clipped_to_grid=bool(used_age != age),
                   teff=teff, luminosity_watts=luminosity,
                   log_luminosity_lsun=float(np.log10(luminosity / L_SUN_WATTS)),
                   logg=logg, mag_F182M=m182, mag_F200W=m200,
                   mag_F070W=np.nan, mag_F555W=np.nan, mag_F814W=np.nan)
        if TEFF_MIN_K <= teff <= TEFF_MAX_K:
            counters['n_in_temperature_band'] += 1
            # Pair 2: F070W/F200W. Retain base/HR even on auxiliary failure.
            aux1 = safe_interpolate_pair(used_age, mass, FILTER_PAIR_KEYS[1])
            if aux1 is not None:
                L2, T2, g2, m070, m200b = map(float, aux1)
                physical_ok = (np.isclose(luminosity, L2, rtol=1e-6)
                               and np.isclose(teff, T2, rtol=1e-6)
                               and np.isclose(logg, g2, rtol=1e-6, atol=1e-7)
                               and np.isclose(m200, m200b, rtol=1e-7, atol=1e-5))
                if physical_ok:
                    row['mag_F070W'] = m070
                else:
                    counters['auxiliary_pair_failed'] += 1
            else:
                counters['auxiliary_pair_failed'] += 1
            # Pair 3: HST F555W/F814W.
            aux2 = safe_interpolate_pair(used_age, mass, FILTER_PAIR_KEYS[2])
            if aux2 is not None:
                L3, T3, g3, m555, m814 = map(float, aux2)
                physical_ok = (np.isclose(luminosity, L3, rtol=1e-6)
                               and np.isclose(teff, T3, rtol=1e-6)
                               and np.isclose(logg, g3, rtol=1e-6, atol=1e-7))
                if physical_ok:
                    row['mag_F555W'], row['mag_F814W'] = m555, m814
                else:
                    counters['auxiliary_pair_failed'] += 1
            else:
                counters['auxiliary_pair_failed'] += 1
        rows.append(row)
        counters['retained'] += 1
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS), counters


def load_finite_table(sim_path, requested_time_myr):
    path = os.path.abspath(str(sim_path))
    if not path.endswith('/'):
        path += '/'
    snapshot = Reader.read_snapshot(path, time=float(requested_time_myr))
    snapshot.to_physical()
    actual_time = float(snapshot.time)
    if abs(actual_time - requested_time_myr) > SNAPSHOT_TIME_TOLERANCE_MYR:
        raise RuntimeError(
            f'Requested t={requested_time_myr:g} Myr but nearest snapshot '
            f'is at t={actual_time:g} Myr; refusing to mislabel the snapshot'
        )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero encountered in log10',
                                category=RuntimeWarning)
        table = converter.to_spicea_table(snapshot)
    return table, actual_time


def catalog_paths(sigma, mass, time_myr):
    root = CACHE_DIR / SIGMA_DIRS[sigma] / f'M{mass}' / 'catalogs'
    root.mkdir(parents=True, exist_ok=True)
    stem = f't{float(time_myr):04.1f}myr'
    return root / f'catalog_{stem}.csv.gz', root / f'accounting_{stem}.json'


def load_or_build_catalog(sigma, mass, time_myr):
    csv_path, json_path = catalog_paths(sigma, mass, time_myr)
    if csv_path.exists() and json_path.exists() and not RECOMPUTE_INTERPOLATION:
        with json_path.open() as f:
            accounting = json.load(f)
        if accounting.get('cache_signature') == CACHE_SIGNATURE:
            df = pd.read_csv(csv_path)
            if set(CATALOG_COLUMNS).issubset(df.columns):
                print(f'  cached: Sigma={sigma:g}, M={mass}, t={time_myr:g}, N={len(df)}')
                return df, accounting
        print(f'  stale cache, rebuilding: {csv_path}')

    table, actual_time = load_finite_table(SIMULATION_PATHS[(sigma, mass)], time_myr)
    masses = np.asarray(table['mass'], float)
    ages = np.asarray(table['age'], float)
    input_age_spread = (
        float(np.nanpercentile(ages, 95) - np.nanpercentile(ages, 5))
        if np.isfinite(ages).any() else np.nan
    )
    df, counters = interpolate_population(masses, ages)
    accounting = dict(
        cache_signature=CACHE_SIGNATURE, sigma_cloud_g_cm2=sigma,
        cluster_mass_msun=mass, epsilon_ff=EPSILON_FF, seed=SEED,
        requested_time_myr=float(time_myr), actual_snapshot_time_myr=actual_time,
        age_spread_input_p95_minus_p05_myr=input_age_spread,
        **counters,
    )
    df.to_csv(csv_path, index=False, compression='gzip')
    json_path.write_text(json.dumps(accounting, indent=2))
    print(f'  built: Sigma={sigma:g}, M={mass}, t={time_myr:g}, '
          f'N={len(df)}/{len(table)}, clipped={counters["n_clipped_young"]}, '
          f'aux failures={counters["auxiliary_pair_failed"]}', flush=True)
    return df, accounting


def catalog_xy(df, diagram):
    if diagram.kind == 'hr':
        return (df['teff'].to_numpy(float),
                df['log_luminosity_lsun'].to_numpy(float))
    return (df[f'mag_{diagram.blue}'].to_numpy(float)
            - df[f'mag_{diagram.red}'].to_numpy(float),
            df[f'mag_{diagram.y_filter}'].to_numpy(float))


# %% [markdown]
# ## Spread measurement, coverage accounting, and per-snapshot processing

# %%
def quartile_tail_spread(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    q25, q75 = np.quantile(v, [0.25, 0.75])
    lower, upper = v[v <= q25], v[v >= q75]
    if not len(lower) or not len(upper):
        return np.nan, np.nan, np.nan
    lo, hi = float(lower.mean()), float(upper.mean())
    return hi - lo, lo, hi


def measure(df, diagram, sigma, mass, time_myr):
    x, y = catalog_xy(df, diagram)
    teff = df['teff'].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(teff)
    teff_ok = finite & (teff >= TEFF_MIN_K) & (teff <= TEFF_MAX_K)
    edges = STATIC_BINS[diagram.key]
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_x = ((x >= lo) & (x <= hi)) if i == N_BINS - 1 else ((x >= lo) & (x < hi))
        chosen = teff_ok & in_x
        y_bin = y[chosen]
        spread = lower = upper = np.nan
        if len(y_bin) >= MIN_STARS_PER_BIN:
            spread, lower, upper = quartile_tail_spread(y_bin)
        bins.append(dict(
            sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass,
            time_myr=time_myr, diagram=diagram.key, bin_index=i,
            x_low=float(lo), x_high=float(hi),
            n_in_x_before_teff_cut=int((finite & in_x).sum()),
            n_selected=int(chosen.sum()), lower_tail_mean_y=lower,
            upper_tail_mean_y=upper, bin_spread=spread,
            valid_bin=bool(np.isfinite(spread)),
        ))
    valid = [r['bin_spread'] for r in bins if r['valid_bin']]
    retained_ages = df['age_myr'].to_numpy(float)
    retained_ages = retained_ages[np.isfinite(retained_ages)]
    summary = dict(
        sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass, epsilon_ff=EPSILON_FF,
        seed=SEED, time_myr=time_myr, diagram=diagram.key,
        spread_metric=float(np.mean(valid)) if valid else np.nan,
        n_valid_bins=len(valid), n_total_bins=N_BINS,
        full_coverage=bool(len(valid) == N_BINS),
        n_catalog_rows=len(df), n_finite_diagram=int(finite.sum()),
        n_in_temperature_band=int(teff_ok.sum()),
        n_in_fixed_bins=int((teff_ok & (x >= edges[0]) & (x <= edges[-1])).sum()),
        age_spread_retained_p95_minus_p05_myr=(
            float(np.percentile(retained_ages, 95) - np.percentile(retained_ages, 5))
            if len(retained_ages) else np.nan
        ),
    )
    return summary, bins


metrics, bin_rows, accounting_rows = [], [], []
for (sigma, mass), sim_path in SIMULATION_PATHS.items():
    print('\n' + '=' * 70)
    print(f'Sigma_cloud={sigma:g} g/cm^2; M_cl={mass} Msun')
    for t in ANALYSIS_TIMES_MYR:
        t = float(t)
        try:
            df, accounting = load_or_build_catalog(sigma, mass, t)
            accounting['status'] = 'success'
            accounting['error'] = ''
        except Exception as exc:
            print(f'  FAILED Sigma={sigma:g}, M={mass}, t={t:g}: '
                  f'{type(exc).__name__}: {exc}', flush=True)
            df = pd.DataFrame(columns=CATALOG_COLUMNS)
            accounting = dict(
                sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass,
                epsilon_ff=EPSILON_FF, seed=SEED, requested_time_myr=t,
                actual_snapshot_time_myr=np.nan, status='failed', error=repr(exc),
                n_input=0, retained=0, n_clipped_young=0,
            )
        accounting_rows.append(accounting)
        for diagram in DIAGRAMS:
            row, detail = measure(df, diagram, sigma, mass, t)
            row['snapshot_status'] = accounting['status']
            row['actual_snapshot_time_myr'] = accounting.get('actual_snapshot_time_myr', np.nan)
            row['n_input_systems'] = accounting.get('n_input', 0)
            row['n_clipped_young'] = accounting.get('n_clipped_young', 0)
            row['age_spread_input_p95_minus_p05_myr'] = accounting.get(
                'age_spread_input_p95_minus_p05_myr', np.nan
            )
            metrics.append(row)
            bin_rows.extend(detail)
        del df
    gc.collect()

DF_METRICS = pd.DataFrame(metrics)
DF_BINS = pd.DataFrame(bin_rows)
DF_ACCOUNTING = pd.DataFrame(accounting_rows)
DF_METRICS.to_csv(OUTPUT_DIR / 'spread_metrics_by_sigma_mass_time_diagram.csv', index=False)
DF_BINS.to_csv(OUTPUT_DIR / 'spread_metrics_per_bin.csv', index=False)
DF_ACCOUNTING.to_csv(OUTPUT_DIR / 'interpolation_accounting.csv', index=False)

coverage = (DF_METRICS.groupby(['sigma_cloud_g_cm2', 'cluster_mass_msun', 'diagram'],
                               as_index=False)
            .agg(n_epochs=('time_myr', 'size'),
                 n_epochs_with_metric=('spread_metric', 'count'),
                 n_full_coverage_epochs=('full_coverage', 'sum'),
                 min_valid_bins=('n_valid_bins', 'min'),
                 median_valid_bins=('n_valid_bins', 'median')))
coverage.to_csv(OUTPUT_DIR / 'metric_coverage_summary.csv', index=False)
print('\nMetric coverage:')
print(coverage.to_string(index=False))


# %% [markdown]
# ## Formation histories: verify the proposed physical mechanism
#
# The birth/insertion times from gradual.97 (converter.get_formation) are
# independent of photometric interpolation. Snapshot-0 components are included
# at t=0 and deduplicated by persistent NAME. The CDF is component-weighted;
# a primordial binary contributes two stellar components.

# %%
def formation_history(sim_path):
    path = os.path.abspath(str(sim_path))
    if not path.endswith('/'):
        path += '/'
    names, times = converter.get_formation(path)
    names = np.asarray(names, int)
    times = np.asarray(times, float)
    if len(names) != len(times):
        raise RuntimeError('Formation NAME and time arrays differ in length')
    lookup = {}
    for name, t0 in zip(names, times):
        if int(name) in lookup and not np.isclose(lookup[int(name)], t0, atol=1e-9):
            raise RuntimeError(f'Conflicting formation times for NAME {name}')
        lookup[int(name)] = float(t0)
    try:
        snap0 = Reader.read_snapshot(path, snapshot=0)
    except TypeError:
        snap0 = Reader.read_snapshot(path, 0)
    snap0.to_physical()
    for name in np.asarray(snap0.stars.name, int):
        if int(name) in lookup and not np.isclose(lookup[int(name)], 0.0, atol=1e-8):
            raise RuntimeError(f'Snapshot-0 NAME {name} has nonzero formation time')
        lookup[int(name)] = 0.0
    t_birth = np.sort(np.asarray(list(lookup.values()), float))
    if not len(t_birth) or not np.all(np.isfinite(t_birth)):
        raise RuntimeError('No valid formation times')
    if t_birth.min() < -1e-8:
        raise RuntimeError('Negative formation times')
    return t_birth


FORMATION_TIMES = {}
formation_summary_rows = []
for (sigma, mass), sim_path in SIMULATION_PATHS.items():
    try:
        times = formation_history(sim_path)
        FORMATION_TIMES[(sigma, mass)] = times
        formation_summary_rows.append(dict(
            sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass,
            n_components=len(times), t_first_myr=float(times[0]),
            t_p05_myr=float(np.percentile(times, 5)),
            t_p50_myr=float(np.percentile(times, 50)),
            t_p95_myr=float(np.percentile(times, 95)),
            t_last_myr=float(times[-1]),
            formation_duration_myr=float(times[-1] - times[0]),
            formation_duration_p05_p95_myr=float(np.percentile(times, 95)
                                                 - np.percentile(times, 5)),
            status='success', error='',
        ))
    except Exception as exc:
        print(f'WARNING: formation history unavailable Sigma={sigma}, M={mass}: {exc}')
        formation_summary_rows.append(dict(
            sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass,
            status='failed', error=repr(exc),
        ))
DF_FORMATION = pd.DataFrame(formation_summary_rows)
DF_FORMATION.to_csv(OUTPUT_DIR / 'formation_duration_by_sigma_mass.csv', index=False)
print('\nFormation histories:')
print(DF_FORMATION.to_string(index=False))


# %% [markdown]
# ## Publication figure 1: F182M-F200W CMD evolution, 3 masses x 6 epochs
#
# Both Sigma figures use the same axes. Points are intentionally small and
# rasterized in the vector PDF; text and isochrone lines remain vector.

# %%
CMD_DIAGRAM = DIAGRAM_BY_KEY['f182m_f200w']


def get_catalog_for_plot(sigma, mass, t):
    csv_path, json_path = catalog_paths(sigma, mass, t)
    if not (csv_path.exists() and json_path.exists()):
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    try:
        metadata = json.loads(json_path.read_text())
        if metadata.get('cache_signature') != CACHE_SIGNATURE:
            return pd.DataFrame(columns=CATALOG_COLUMNS)
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f'WARNING: cannot read plot catalog {csv_path}: {exc}')
        return pd.DataFrame(columns=CATALOG_COLUMNS)


def cmd_shared_limits():
    x_lows, x_highs, y_lows, y_highs = [], [], [], []
    for sigma in SIGMA_DIRS:
        for mass in MASS_DIRS:
            for t in DISPLAY_TIMES_MYR:
                df = get_catalog_for_plot(sigma, mass, t)
                if df.empty:
                    continue
                x, y = catalog_xy(df, CMD_DIAGRAM)
                good = np.isfinite(x) & np.isfinite(y)
                if good.sum() < 3:
                    continue
                x_lows.append(float(np.percentile(x[good], 0.1)))
                x_highs.append(float(np.percentile(x[good], 99.9)))
                y_lows.append(float(np.percentile(y[good], 0.1)))
                y_highs.append(float(np.percentile(y[good], 99.9)))
    if not x_lows:
        raise RuntimeError('No CMD catalogs available for figure 1')
    xmin = min(-0.05, min(x_lows) - 0.02)
    xmax = max(0.45, max(x_highs) + 0.02)
    ybright = min(-2.0, min(y_lows) - 0.5)
    yfaint = max(20.0, max(y_highs) + 0.5)
    return (xmin, xmax), (yfaint, ybright)  # reversed magnitude axis


CMD_XLIM, CMD_YLIM = cmd_shared_limits()
print('Shared CMD limits:', CMD_XLIM, CMD_YLIM)


def shade_cmd_bins(ax):
    edges = STATIC_BINS[CMD_DIAGRAM.key]
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 0:
            ax.axvspan(left, right, color='#9EADBC', alpha=0.065, zorder=0)
        ax.axvline(left, color='0.70', lw=0.35, alpha=0.35, zorder=0)
    ax.axvline(edges[-1], color='0.70', lw=0.35, alpha=0.35, zorder=0)


def metric_at(sigma, mass, t, diagram_key):
    sub = DF_METRICS[
        np.isclose(DF_METRICS['sigma_cloud_g_cm2'], sigma)
        & (DF_METRICS['cluster_mass_msun'] == mass)
        & np.isclose(DF_METRICS['time_myr'], t)
        & (DF_METRICS['diagram'] == diagram_key)
    ]
    return sub.iloc[0] if len(sub) == 1 else None


def plot_cmd_panel(ax, sigma, mass, t):
    shade_cmd_bins(ax)
    iso_idx = int(np.argmin(np.abs(ISO_GRID.ages_myr - t)))
    xi, yi, _ = isochrone_xy(ISO_GRID.isochrones[iso_idx], CMD_DIAGRAM)
    good_iso = np.isfinite(xi) & np.isfinite(yi)
    ax.plot(xi[good_iso], yi[good_iso], color='0.22', lw=0.85,
            alpha=0.72, zorder=2)

    df = get_catalog_for_plot(sigma, mass, t)
    if not df.empty:
        x, y = catalog_xy(df, CMD_DIAGRAM)
        teff = df['teff'].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        eligible = good & (teff >= TEFF_MIN_K) & (teff <= TEFF_MAX_K)
        ax.scatter(x[good], y[good], s=1.0, color='0.62', alpha=0.13,
                   linewidths=0, rasterized=True, zorder=3)
        alpha = 0.58 if eligible.sum() < 250 else (0.38 if eligible.sum() < 1500 else 0.23)
        ax.scatter(x[eligible], y[eligible], s=2.1,
                   color=MASS_COLORS[mass], alpha=alpha,
                   linewidths=0, rasterized=True, zorder=4)
    else:
        ax.text(0.5, 0.5, 'Snapshot unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='0.35')

    metric = metric_at(sigma, mass, t, CMD_DIAGRAM.key)
    if metric is not None and np.isfinite(metric['spread_metric']):
        label = f"$S$ = {metric['spread_metric']:.2f} mag\n{int(metric['n_valid_bins'])}/6 bins"
    else:
        label = '$S$ = n/a'
    ax.text(0.97, 0.04, label, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.0,
            bbox=dict(facecolor='white', alpha=0.78,
                      edgecolor='none', boxstyle='round,pad=0.15'), zorder=6)
    ax.set_xlim(*CMD_XLIM)
    ax.set_ylim(*CMD_YLIM)
    ax.grid(alpha=0.11, lw=0.5)
    ax.tick_params(labelsize=7, length=2)


for sigma, sigma_dir in SIGMA_DIRS.items():
    masses = list(MASS_DIRS)
    fig, axes = plt.subplots(3, len(DISPLAY_TIMES_MYR), figsize=(18.0, 9.0),
                             sharex=True, sharey=True)
    for r, mass in enumerate(masses):
        for c, t in enumerate(DISPLAY_TIMES_MYR):
            ax = axes[r, c]
            plot_cmd_panel(ax, sigma, mass, t)
            if r == 0:
                ax.set_title(f'$t$ = {t:g} Myr', fontsize=10, pad=7)
            if c == 0:
                ax.set_ylabel(rf'$M_{{\rm cl}}={mass:,}\ M_\odot$' + '\nF200W',
                              fontsize=9)
            if r == len(masses) - 1:
                ax.set_xlabel('F182M - F200W', fontsize=9)
    fig.suptitle(
        r'JWST F182M $-$ F200W vs. F200W: cluster-mass evolution'
        + '\n' + rf'$\Sigma_{{\rm cloud}}={sigma:g}\ \mathrm{{g\,cm^{{-2}}}}$; '
        + rf'$\epsilon_{{\rm ff}}={EPSILON_FF:g}$; seed {SEED}; '
        + rf'$d={DISTANCE_PC:g}\ \mathrm{{pc}}$',
        y=0.994, fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], color='0.22', lw=1, label='Isochrone at snapshot age'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.62',
               markeredgecolor='none', markersize=4, label='All interpolated systems'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#3467A7',
               markeredgecolor='none', markersize=4, label='Selected: 3000-3600 K'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.933), fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.07, top=0.865,
                        wspace=0.06, hspace=0.10)
    save_figure(fig, f'cmd_evolution_f182m_f200w_{sigma_dir}')


# %% [markdown]
# ## Publication figure 2: four spread metrics across both Sigmas
#
# Solid dots: all six bins valid. Hollow dots: only some bins valid. A curve
# through partial-bin estimates is descriptive, not a complete-bin estimate.
# Faint vertical dotted lines mark the final formation/insertion time from
# the N-body formation record. One seed => no statistical uncertainty band.

# %%
def draw_metric_curves(ax, sigma, diagram, show_ylabel=True):
    for mass in MASS_DIRS:
        sub = DF_METRICS[
            np.isclose(DF_METRICS['sigma_cloud_g_cm2'], sigma)
            & (DF_METRICS['cluster_mass_msun'] == mass)
            & (DF_METRICS['diagram'] == diagram.key)
        ].sort_values('time_myr')
        t = sub['time_myr'].to_numpy(float)
        y = sub['spread_metric'].to_numpy(float)
        nvalid = sub['n_valid_bins'].to_numpy(int)
        good = np.isfinite(y)
        full = good & (nvalid == N_BINS)
        partial = good & (nvalid < N_BINS)
        color = MASS_COLORS[mass]
        ax.plot(t, y, color=color, lw=1.65, alpha=0.92, zorder=2)
        ax.scatter(t[full], y[full], color=color, s=14, zorder=3)
        ax.scatter(t[partial], y[partial], facecolors='white',
                   edgecolors=color, linewidths=0.9, s=18, zorder=4)
        ft = FORMATION_TIMES.get((sigma, mass))
        if ft is not None and 0.0 <= ft[-1] <= ANALYSIS_TIMES_MYR[-1]:
            ax.axvline(ft[-1], color=color, ls=':', lw=0.85,
                       alpha=0.38, zorder=1)
    ax.set_title(diagram.title, pad=7)
    ax.set_xlim(ANALYSIS_TIMES_MYR[0], ANALYSIS_TIMES_MYR[-1])
    ax.set_xlabel('Cluster time [Myr]')
    if show_ylabel:
        ax.set_ylabel('Spread [dex]' if diagram.kind == 'hr' else 'Spread [mag]')
    ax.grid(alpha=0.20, lw=0.6)


mass_handles = [
    Line2D([0], [0], color=MASS_COLORS[m], lw=2,
           label=rf'$M_{{\rm cl}}={m:,}\ M_\odot$')
    for m in MASS_DIRS
]
extra_handles = [
    Line2D([0], [0], marker='o', linestyle='none', color='0.3', markersize=4,
           label='6/6 bins valid'),
    Line2D([0], [0], marker='o', linestyle='none', color='0.3',
           markerfacecolor='white', markersize=4, label='<6 bins valid'),
    Line2D([0], [0], linestyle=':', color='0.4', lw=1,
           label='Last stellar formation'),
]

fig, axes = plt.subplots(2, len(DIAGRAMS), figsize=(17.0, 8.1), sharex=True)
for r, sigma in enumerate(SIGMA_DIRS):
    for c, diagram in enumerate(DIAGRAMS):
        draw_metric_curves(axes[r, c], sigma, diagram)
    axes[r, 0].text(-0.27, 0.5,
                    rf'$\Sigma_{{\rm cloud}}={sigma:g}$' + '\n' +
                    r'$\mathrm{g\,cm^{-2}}$',
                    transform=axes[r, 0].transAxes, ha='center', va='center',
                    rotation=90, fontsize=10)
fig.suptitle('Cluster-mass dependence of CMD and H-R spread\n'
             + rf'$\epsilon_{{\rm ff}}={EPSILON_FF:g}$; seed {SEED}',
             y=0.995, fontsize=14)
fig.legend(handles=mass_handles + extra_handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.912), ncol=6, frameon=False, fontsize=8)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.08, top=0.83,
                    wspace=0.29, hspace=0.31)
save_figure(fig, 'spread_evolution_both_sigmas_2x4')

for sigma, sigma_dir in SIGMA_DIRS.items():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), sharex=True)
    for ax, diagram in zip(axes.flat, DIAGRAMS):
        draw_metric_curves(ax, sigma, diagram)
    fig.suptitle('Cluster-mass dependence of CMD and H-R spread\n'
                 + rf'$\Sigma_{{\rm cloud}}={sigma:g}\ \mathrm{{g\,cm^{{-2}}}}$; '
                 + rf'$\epsilon_{{\rm ff}}={EPSILON_FF:g}$; seed {SEED}',
                 y=0.995, fontsize=13)
    fig.legend(handles=mass_handles + extra_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.83,
                        wspace=0.28, hspace=0.28)
    save_figure(fig, f'spread_evolution_{sigma_dir}_2x2')


# %% [markdown]
# ## Formation CDF: does increasing M_cl actually extend star formation?

# %%
fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
for ax, sigma in zip(axes, SIGMA_DIRS):
    for mass in MASS_DIRS:
        birth = FORMATION_TIMES.get((sigma, mass))
        if birth is None:
            continue
        y = np.arange(1, len(birth) + 1, dtype=float) / len(birth)
        ax.step(birth, y, where='post', color=MASS_COLORS[mass], lw=1.7,
                label=rf'$M_{{\rm cl}}={mass:,}\ M_\odot$')
        ax.plot(birth[-1], 1.0, marker='o', ms=4, color=MASS_COLORS[mass])
    ax.set_title(rf'$\Sigma_{{\rm cloud}}={sigma:g}\ \mathrm{{g\,cm^{{-2}}}}$')
    ax.set_ylabel('Cumulative fraction formed')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.20)
axes[-1].set_xlabel('Stellar formation/insertion time [Myr]')
axes[0].legend(frameon=False, ncol=3, loc='lower right')
fig.suptitle('Stellar formation histories by cluster mass\n'
             'Component-weighted; seed 00, '
             + rf'$\epsilon_{{\rm ff}}={EPSILON_FF:g}$', y=0.995, fontsize=13)
fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.10, hspace=0.32)
save_figure(fig, 'formation_histories_by_cluster_mass')


# %% [markdown]
# ## Peak spread vs cluster mass (descriptive; one seed per point)

# %%
peak_rows = []
for sigma in SIGMA_DIRS:
    for mass in MASS_DIRS:
        for diagram in DIAGRAMS:
            sub = DF_METRICS[
                np.isclose(DF_METRICS['sigma_cloud_g_cm2'], sigma)
                & (DF_METRICS['cluster_mass_msun'] == mass)
                & (DF_METRICS['diagram'] == diagram.key)
                & np.isfinite(DF_METRICS['spread_metric'])
            ]
            if len(sub):
                peak = sub.loc[sub['spread_metric'].idxmax()]
                max_spread = float(peak['spread_metric'])
                peak_time = float(peak['time_myr'])
                nvalid = int(peak['n_valid_bins'])
            else:
                max_spread, peak_time, nvalid = np.nan, np.nan, 0
            ft = FORMATION_TIMES.get((sigma, mass))
            peak_rows.append(dict(
                sigma_cloud_g_cm2=sigma, cluster_mass_msun=mass,
                diagram=diagram.key, peak_spread=max_spread,
                time_of_peak_myr=peak_time, n_valid_bins_at_peak=nvalid,
                formation_end_myr=float(ft[-1]) if ft is not None else np.nan,
            ))
DF_PEAK = pd.DataFrame(peak_rows)
DF_PEAK.to_csv(OUTPUT_DIR / 'peak_spread_vs_cluster_mass.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
for ax, diagram in zip(axes.flat, DIAGRAMS):
    for sigma, ls in ((0.1, '-'), (1.0, '--')):
        sub = DF_PEAK[(DF_PEAK['diagram'] == diagram.key)
                      & np.isclose(DF_PEAK['sigma_cloud_g_cm2'], sigma)]
        sub = sub.sort_values('cluster_mass_msun')
        ax.plot(sub['cluster_mass_msun'], sub['peak_spread'], ls=ls,
                color='0.48', lw=1.0, alpha=0.65)
        for row in sub.itertuples():
            full = row.n_valid_bins_at_peak == N_BINS
            ax.scatter(row.cluster_mass_msun, row.peak_spread,
                       s=48, marker='o' if sigma == 0.1 else 's',
                       facecolor=MASS_COLORS[row.cluster_mass_msun] if full else 'white',
                       edgecolor=MASS_COLORS[row.cluster_mass_msun],
                       linewidth=1.4, zorder=4)
    ax.set_xscale('log')
    ax.set_xticks(list(MASS_DIRS))
    ax.set_xticklabels(['300', '3,000', '30,000'])
    ax.set_title(diagram.title)
    ax.set_xlabel(r'$M_{\rm cl}\ [M_\odot]$')
    ax.set_ylabel('Peak spread [dex]' if diagram.kind == 'hr' else 'Peak spread [mag]')
    ax.grid(alpha=0.2)
fig.suptitle('Peak spread vs. cluster mass (one realization per mass)\n'
             'Open symbols: peak uses fewer than six valid bins', y=0.994, fontsize=13)
fig.legend(handles=[Line2D([0], [0], marker='o', ls='none', color='0.3',
                           label=r'$\Sigma_{\rm cloud}=0.1$'),
                    Line2D([0], [0], marker='s', ls='none', color='0.3',
                           label=r'$\Sigma_{\rm cloud}=1.0$')],
           loc='upper center', bbox_to_anchor=(0.5, 0.915), ncol=2,
           frameon=False)
fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.09,
                    wspace=0.30, hspace=0.31)
save_figure(fig, 'peak_spread_vs_cluster_mass')

# %% [markdown]
# ## Peak spread vs measured formation duration
#
# This tests the proposed intermediary (longer star formation), rather than
# inferring it solely from the mass labels. Six points total per diagram.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
for ax, diagram in zip(axes.flat, DIAGRAMS):
    for row in DF_PEAK[DF_PEAK['diagram'] == diagram.key].itertuples():
        if not (np.isfinite(row.formation_end_myr) and np.isfinite(row.peak_spread)):
            continue
        complete = row.n_valid_bins_at_peak == N_BINS
        ax.scatter(row.formation_end_myr, row.peak_spread, s=53,
                   marker='o' if row.sigma_cloud_g_cm2 == 0.1 else 's',
                   facecolor=MASS_COLORS[row.cluster_mass_msun] if complete else 'white',
                   edgecolor=MASS_COLORS[row.cluster_mass_msun],
                   linewidth=1.4, zorder=3)
    ax.set_title(diagram.title)
    ax.set_xlabel('Last stellar formation time [Myr]')
    ax.set_ylabel('Peak spread [dex]' if diagram.kind == 'hr' else 'Peak spread [mag]')
    ax.grid(alpha=0.2)
fig.suptitle('Peak spread vs. star-formation duration (seed 00)\n'
             'Mass shown by color; Sigma by marker shape', y=0.994, fontsize=13)
fig.legend(handles=mass_handles + [
    Line2D([0], [0], marker='o', ls='none', color='0.3',
           label=r'$\Sigma_{\rm cloud}=0.1$'),
    Line2D([0], [0], marker='s', ls='none', color='0.3',
           label=r'$\Sigma_{\rm cloud}=1.0$')],
    loc='upper center', bbox_to_anchor=(0.5, 0.915), ncol=5, frameon=False)
fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.09,
                    wspace=0.30, hspace=0.31)
save_figure(fig, 'peak_spread_vs_formation_duration')


# %% [markdown]
# ## Metadata and output manifest

# %%
metadata = dict(
    comparison='cluster mass at two cloud surface densities',
    epsilon_ff=EPSILON_FF, seed=SEED,
    sigma_cloud_g_cm2=list(SIGMA_DIRS),
    cluster_masses_msun=list(MASS_DIRS),
    simulation_paths={f'{s}:{m}': str(p) for (s, m), p in SIMULATION_PATHS.items()},
    distance_pc=DISTANCE_PC, aks=AKS,
    iso_cache_dir=str(ISO_CACHE_DIR), cache_signature=CACHE_SIGNATURE,
    isochrone_ages_myr=ISO_AGES_MYR.tolist(),
    analysis_times_myr=ANALYSIS_TIMES_MYR.tolist(),
    display_times_myr=list(DISPLAY_TIMES_MYR),
    temperature_range_k=[TEFF_MIN_K, TEFF_MAX_K],
    n_bins=N_BINS, min_stars_per_bin=MIN_STARS_PER_BIN,
    fixed_bins={k: v.tolist() for k, v in STATIC_BINS.items()},
    reference_bin_age_myr=REFERENCE_BIN_AGE_MYR,
    diagrams=[asdict(d) for d in DIAGRAMS],
    filter_obsmodes=FILTER_OBSMODES,
    filter_pairs=[list(p) for p in FILTER_PAIR_NAMES],
    physical_population='one primary per current converter system; companions not flux-combined',
    metric_definition='mean of valid per-bin (upper 25% mean - lower 25% mean)',
    caveats=[
        'One seed per mass and Sigma: trends are descriptive, not ensemble estimates.',
        'Peak spread need not be linearly proportional to cluster mass.',
        'Fewer than six valid bins changes which regions contribute to the global mean.',
        'Stellar ages younger than 1 Myr are clipped to 1 Myr for interpolation.',
        'Formation history is component-weighted and independent of interpolation.',
        'Photometric completeness, extinction scatter, and unresolved companion flux are not modeled.',
    ],
)
(OUTPUT_DIR / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
manifest = pd.DataFrame([
    ('cmd_evolution_f182m_f200w_sigma0p1', '3x6 F182M-F200W CMD evolution, Sigma=0.1'),
    ('cmd_evolution_f182m_f200w_sigma1p0', '3x6 F182M-F200W CMD evolution, Sigma=1.0'),
    ('spread_evolution_both_sigmas_2x4', 'H-R and 3 CMD spreads, two Sigma rows'),
    ('spread_evolution_sigma0p1_2x2', 'Four spread metrics at Sigma=0.1'),
    ('spread_evolution_sigma1p0_2x2', 'Four spread metrics at Sigma=1.0'),
    ('formation_histories_by_cluster_mass', 'Formation-time CDF by Sigma and mass'),
    ('peak_spread_vs_cluster_mass', 'Peak metric amplitude vs mass'),
    ('peak_spread_vs_formation_duration', 'Peak metric amplitude vs measured SF duration'),
    ('spread_metrics_by_sigma_mass_time_diagram.csv', 'Per-snapshot spread metrics'),
    ('spread_metrics_per_bin.csv', 'Per-bin star counts and spread'),
    ('metric_coverage_summary.csv', 'Coverage and valid-bin diagnostics'),
    ('interpolation_accounting.csv', 'Per-snapshot interpolation success and clipping'),
    ('formation_duration_by_sigma_mass.csv', 'Formation times and durations'),
    ('peak_spread_vs_cluster_mass.csv', 'Peak metrics, times, and valid bins'),
    ('fixed_bin_edges.csv', 'Fixed CMD/H-R bin edges'),
    ('isochrone_coverage.csv', 'Isochrone build/cache diagnostics'),
    ('analysis_metadata.json', 'Reproducibility configuration'),
], columns=['stem_or_filename', 'description'])
manifest.to_csv(OUTPUT_DIR / 'output_manifest.csv', index=False)

print('\n' + '=' * 70)
print('Completed cluster-mass comparison')
print('Output directory:', OUTPUT_DIR.resolve())
print('Cache directory:', CACHE_DIR.resolve())
print('Main metrics:', OUTPUT_DIR / 'spread_metrics_by_sigma_mass_time_diagram.csv')
print('Formation duration:', OUTPUT_DIR / 'formation_duration_by_sigma_mass.csv')
print('NOTE: test monotonicity/proportionality from the CSV; it is not imposed.')
