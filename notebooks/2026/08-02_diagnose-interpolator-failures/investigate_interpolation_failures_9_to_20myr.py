# %% [markdown]
# # Diagnose rare interpolation failures
#
# This Jupytext-compatible script investigates systems rejected by the
# generalized ``interpolator.py`` in the existing multi-diagram analysis.
#
# It reproduces the same:
#
# - merged Baraffe--Pisa--Ekstrom--Parsec model;
# - 1--20 Myr isochrone grid in 0.5 Myr increments;
# - JWST + HST filter set;
# - sigma=0.1, epsilon_ff=0.03, seed 00 simulation.
#
# For every rejected system, it records:
#
# - requested mass and age;
# - the two age-grid indices selected by ``findIsoIdx``;
# - the mass range of each bracketing isochrone;
# - whether the mass is outside either isochrone;
# - the nearest mass-grid point and proposed neighbor;
# - duplicate or zero-width mass intervals;
# - missing/non-finite physical or photometric columns;
# - the exact exception raised by each interpolation stage;
# - whether failure occurs in mass interpolation, age interpolation,
#   result-shape validation, or finite-value validation.
#
# By default, it checks every snapshot from 9.0 through 20.0 Myr in
# 0.5 Myr increments, covering the range where failures begin and grow.

# %%
from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from astropy.table import Column, Table

from spisea import atmospheres, reddening, synthetic
from nbody6tools import Reader
from nbody62spisea import converter

sys.path.append("/home/wyz5rge/synthetic-cmd-dev/cmd_generator")
import interpolator  # noqa: E402


# %% [markdown]
# ## Configuration

# %%
UPDATED_MERGED_ROOT = Path(
    "/home/wyz5rge/SPISEA/evolution/merged/"
    "baraffe_pisa_ekstrom_parsec/"
)

SIMULATION_PATH = Path(
    "/standard/Tan_JC/backup_protoclusters/multiples/M3000new/"
    "sigma0p1/fiducial/sfe_ff003/00"
)

# Reuse the same cache as the original analysis.
ISO_CACHE_DIR = Path.cwd() / "single_sim_multi_diagram_iso_cache"
OUTPUT_DIR = Path.cwd() / "interpolation_failure_diagnostics"

RESET_ISO_CACHE = False

SNAPSHOT_TIMES_MYR = np.arange(9.0, 20.0 + 0.25, 0.5)

USE_ROTATING_MERGED = False
AKS = 0.0
DISTANCE_PC = 410.0
METALLICITY = 0.0

ATM_FUNC = atmospheres.get_BTSettl_2015_atmosphere
RED_LAW = reddening.RedLawHosek18b()

ISO_AGES_MYR = np.arange(1.0, 20.0 + 0.25, 0.5)
ISO_LOG_AGES = np.log10(ISO_AGES_MYR * 1.0e6)

CLIP_YOUNG_STARS_TO_GRID_MINIMUM = True

FILTER_OBSMODES = {
    "F070W": "jwst,F070W",
    "F140M": "jwst,F140M",
    "F162M": "jwst,F162M",
    "F182M": "jwst,F182M",
    "F200W": "jwst,F200W",
    "F555W": "wfc3,uvis1,f555w",
    "F814W": "wfc3,uvis1,f814w",
}

ALL_FILTERS = list(FILTER_OBSMODES.values())


# %% [markdown]
# ## Merged evolutionary-model reader

# %%
class MergedBaraffePisaEkstromParsecDAT:
    def __init__(self, root_dir: Path | str, rot: bool = False):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.rot = bool(rot)
        self.model_dir = str(self.root_dir)
        self.z_list = [0.015]
        self.z_solar = 0.015
        self.mass_list: list[float] = []

        self.grid_dir = self.root_dir / (
            "z015_rot" if rot else "z015_norot"
        )

        if not self.grid_dir.is_dir():
            raise FileNotFoundError(self.grid_dir)

        self.age_file_map: dict[float, Path] = {}

        for path in sorted(self.grid_dir.glob("iso_*.dat")):
            try:
                log_age = float(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue

            self.age_file_map[round(log_age, 2)] = path

        if not self.age_file_map:
            raise FileNotFoundError(
                f"No iso_*.dat files found in {self.grid_dir}"
            )

        self.age_list = np.array(
            sorted(self.age_file_map),
            dtype=float,
        )

    def isochrone(
        self,
        age: float = 1.0e6,
        metallicity: float = 0.0,
    ) -> Table:
        requested = math.log10(age)

        if requested < self.age_list[0] or requested > self.age_list[-1]:
            raise ValueError(
                f"logAge {requested:.4f} outside merged grid"
            )

        idx = int(
            np.argmin(np.abs(self.age_list - requested))
        )
        selected = float(self.age_list[idx])
        path = self.age_file_map[round(selected, 2)]

        dtype = [
            ("mass", "f8"),
            ("logT", "f8"),
            ("logL", "f8"),
            ("logg", "f8"),
            ("logT_WR", "f8"),
            ("mass_current", "f8"),
            ("phase", "i4"),
            ("model_ref", "U32"),
        ]

        data = np.genfromtxt(
            path,
            comments="#",
            dtype=dtype,
            encoding="utf-8",
        )

        iso = Table(np.atleast_1d(data))

        is_wr = ~np.isclose(
            np.asarray(iso["logT"], dtype=float),
            np.asarray(iso["logT_WR"], dtype=float),
            rtol=0.0,
            atol=1.0e-8,
        )
        iso.add_column(Column(is_wr, name="isWR"))

        iso.meta.update(
            {
                "log_age": selected,
                "log_age_requested": requested,
                "metallicity_in": metallicity,
                "metallicity_act": 0.0,
                "source_file": str(path),
            }
        )

        return iso


# %% [markdown]
# ## Grid helpers

# %%
@dataclass
class IsoGrid:
    ages_myr: np.ndarray
    log_ages: np.ndarray
    isochrones: list
    filter_columns: dict[str, str]


def prepare_directories() -> None:
    if RESET_ISO_CACHE and ISO_CACHE_DIR.exists():
        shutil.rmtree(ISO_CACHE_DIR)

    ISO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    return "".join(
        ch.lower()
        for ch in str(name)
        if ch.isalnum()
    )


def resolve_filter_column(
    colnames: Sequence[str],
    filter_name: str,
) -> str:
    target = normalize_name(filter_name)

    candidates = [
        column
        for column in colnames
        if normalize_name(column).startswith("m")
        and normalize_name(column).endswith(target)
    ]

    instrument = (
        "hst"
        if filter_name in {"F555W", "F814W"}
        else "jwst"
    )

    preferred = [
        column
        for column in candidates
        if instrument in normalize_name(column)
    ]

    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Cannot uniquely resolve {filter_name}; "
        f"candidates={candidates}; available={list(colnames)}"
    )


def build_iso_grid(evo_model) -> IsoGrid:
    isochrones = []
    filter_columns: dict[str, str] | None = None

    for age_myr, log_age in zip(
        ISO_AGES_MYR,
        ISO_LOG_AGES,
    ):
        print(
            f"Building/loading {age_myr:4.1f} Myr isochrone"
        )

        iso = synthetic.IsochronePhot(
            log_age,
            AKS,
            DISTANCE_PC,
            metallicity=METALLICITY,
            evo_model=evo_model,
            atm_func=ATM_FUNC,
            red_law=RED_LAW,
            filters=ALL_FILTERS,
            iso_dir=str(ISO_CACHE_DIR),
        )

        current = {
            name: resolve_filter_column(
                iso.points.colnames,
                name,
            )
            for name in FILTER_OBSMODES
        }

        if filter_columns is None:
            filter_columns = current

            print("Resolved magnitude columns:")
            for key, value in current.items():
                print(f"  {key}: {value}")

        elif current != filter_columns:
            raise RuntimeError(
                "Magnitude-column names changed across ages."
            )

        isochrones.append(iso)

    if filter_columns is None:
        raise RuntimeError(
            "No isochrones were generated."
        )

    return IsoGrid(
        ages_myr=ISO_AGES_MYR.copy(),
        log_ages=ISO_LOG_AGES.copy(),
        isochrones=isochrones,
        filter_columns=filter_columns,
    )


# %%
prepare_directories()

if not SIMULATION_PATH.is_dir():
    raise FileNotFoundError(SIMULATION_PATH)

evo_model = MergedBaraffePisaEkstromParsecDAT(
    UPDATED_MERGED_ROOT,
    rot=USE_ROTATING_MERGED,
)

ISO_GRID = build_iso_grid(evo_model)

FILTER_KEYS = [
    ISO_GRID.filter_columns[name]
    for name in FILTER_OBSMODES
]

print("Filter keys passed to interpolator:")
for name, key in zip(FILTER_OBSMODES, FILTER_KEYS):
    print(f"  {name}: {key}")


# %% [markdown]
# ## Snapshot loading

# %%
def load_cluster_table(
    sim_path: Path | str,
    snapshot_time_myr: float,
) -> Table:
    path = os.path.abspath(str(sim_path))

    if not path.endswith("/"):
        path += "/"

    snapshot = Reader.read_snapshot(
        path,
        time=float(snapshot_time_myr),
    )
    snapshot.to_physical()

    return converter.to_spicea_table(snapshot)


# %% [markdown]
# ## Low-level diagnostic helpers

# %%
def value_is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def inspect_isochrone_mass_step(
    iso,
    mass: float,
    filters: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Inspect the exact mass interpolation geometry in one isochrone."""
    record: dict[str, Any] = {
        f"{label}_inspection_status": "not_started",
    }

    try:
        points = iso.points
        required = [
            "mass",
            "L",
            "Teff",
            "logg",
            *filters,
        ]

        missing = [
            column
            for column in required
            if column not in points.colnames
        ]

        record[f"{label}_missing_columns"] = "|".join(missing)

        if missing:
            record[f"{label}_inspection_status"] = "missing_columns"
            return record

        mass_grid = np.asarray(
            points["mass"],
            dtype=float,
        )

        finite_idx = np.where(
            np.isfinite(mass_grid)
        )[0]

        if len(finite_idx) == 0:
            record[f"{label}_inspection_status"] = "no_finite_mass"
            return record

        finite_mass = mass_grid[finite_idx]

        record[f"{label}_n_points"] = len(points)
        record[f"{label}_n_finite_mass"] = len(finite_idx)
        record[f"{label}_mass_min"] = float(
            np.min(finite_mass)
        )
        record[f"{label}_mass_max"] = float(
            np.max(finite_mass)
        )
        record[f"{label}_mass_outside_grid"] = bool(
            mass < np.min(finite_mass)
            or mass > np.max(finite_mass)
        )

        duplicate_count = int(
            len(finite_mass)
            - len(np.unique(finite_mass))
        )
        record[f"{label}_duplicate_mass_count"] = duplicate_count
        record[f"{label}_mass_grid_monotonic"] = bool(
            np.all(np.diff(finite_mass) >= 0)
        )

        nearest_local = int(
            np.argmin(np.abs(finite_mass - mass))
        )
        s1_idx = int(finite_idx[nearest_local])
        s1_mass = float(mass_grid[s1_idx])

        record[f"{label}_nearest_index"] = s1_idx
        record[f"{label}_nearest_mass"] = s1_mass
        record[f"{label}_exact_mass_match"] = bool(
            s1_mass == mass
        )
        record[f"{label}_nearest_mass_delta"] = float(
            s1_mass - mass
        )

        if s1_mass < mass:
            candidates = finite_idx[finite_idx > s1_idx]
            direction = "higher_index"
            s2_idx = (
                int(candidates[0])
                if len(candidates) > 0
                else None
            )

        elif s1_mass > mass:
            candidates = finite_idx[finite_idx < s1_idx]
            direction = "lower_index"
            s2_idx = (
                int(candidates[-1])
                if len(candidates) > 0
                else None
            )

        else:
            direction = "exact_match"
            s2_idx = None

        record[f"{label}_neighbor_direction"] = direction
        record[f"{label}_neighbor_index"] = s2_idx

        if s2_idx is not None:
            s2_mass = float(mass_grid[s2_idx])
            denom = s2_mass - s1_mass

            record[f"{label}_neighbor_mass"] = s2_mass
            record[f"{label}_mass_denominator"] = denom
            record[f"{label}_zero_mass_denominator"] = bool(
                np.isclose(denom, 0.0)
            )

        else:
            record[f"{label}_neighbor_mass"] = np.nan
            record[f"{label}_mass_denominator"] = np.nan
            record[f"{label}_zero_mass_denominator"] = False

        rows_to_check = [s1_idx]

        if s2_idx is not None:
            rows_to_check.append(s2_idx)

        nonfinite_cells = []

        for row_idx in rows_to_check:
            for column in required[1:]:
                value = points[row_idx][column]

                if not value_is_finite(value):
                    nonfinite_cells.append(
                        f"row={row_idx},column={column},value={value}"
                    )

        record[f"{label}_nonfinite_cells"] = "|".join(
            nonfinite_cells
        )
        record[f"{label}_inspection_status"] = "completed"

    except Exception as exc:
        record[f"{label}_inspection_status"] = "inspection_exception"
        record[f"{label}_inspection_exception_type"] = type(exc).__name__
        record[f"{label}_inspection_exception"] = str(exc)

    return record


def diagnose_interpolation(
    age_myr: float,
    mass: float,
    grid: IsoGrid,
    filters: Sequence[str],
) -> dict[str, Any]:
    """Run and classify every interpolation stage without suppressing errors."""
    record: dict[str, Any] = {
        "age_used_myr": float(age_myr),
        "mass": float(mass),
        "overall_status": "not_started",
        "failure_stage": "",
        "failure_reason": "",
    }

    # Stage 1: age bracketing.
    try:
        age_indices = interpolator.findIsoIdx(
            age_myr,
            grid.log_ages,
        )

        record["findIsoIdx_return"] = str(age_indices)

        if age_indices is None:
            record["overall_status"] = "failed"
            record["failure_stage"] = "age_bracketing"
            record["failure_reason"] = "findIsoIdx_returned_None"
            return record

        a1, a2 = age_indices

        record["age_index_1"] = int(a1)
        record["age_index_2"] = int(a2)
        record["age_grid_1_myr"] = float(
            10.0 ** grid.log_ages[a1] / 1.0e6
        )
        record["age_grid_2_myr"] = float(
            10.0 ** grid.log_ages[a2] / 1.0e6
        )

    except Exception as exc:
        record["overall_status"] = "failed"
        record["failure_stage"] = "age_bracketing_exception"
        record["failure_reason"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        return record

    # Add detailed geometry for both isochrones.
    record.update(
        inspect_isochrone_mass_step(
            grid.isochrones[a1],
            mass,
            filters,
            "iso1",
        )
    )

    if a2 == a1:
        record.update(
            {
                "iso2_inspection_status": "same_as_iso1",
            }
        )
    else:
        record.update(
            inspect_isochrone_mass_step(
                grid.isochrones[a2],
                mass,
                filters,
                "iso2",
            )
        )

    # Stage 2: mass interpolation in first isochrone.
    try:
        s1 = interpolator.isoInterp(
            mass,
            a1,
            grid.isochrones,
            filters,
        )
        record["isoInterp_1_returned_none"] = s1 is None
        record["isoInterp_1_result_length"] = (
            len(s1)
            if s1 is not None
            else 0
        )

        if s1 is None:
            record["overall_status"] = "failed"
            record["failure_stage"] = "mass_interpolation_iso1"
            record["failure_reason"] = classify_mass_failure(
                record,
                "iso1",
            )
            return record

    except Exception as exc:
        record["overall_status"] = "failed"
        record["failure_stage"] = "mass_interpolation_iso1_exception"
        record["failure_reason"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        return record

    # Exact-age case.
    if a1 == a2:
        try:
            result = np.asarray(
                s1[1:],
                dtype=float,
            )
            record["final_result_length"] = len(result)
            record["final_result_all_finite"] = bool(
                np.all(np.isfinite(result))
            )

            if len(result) != 3 + len(filters):
                record["overall_status"] = "failed"
                record["failure_stage"] = "result_shape"
                record["failure_reason"] = "unexpected_result_length"
                return record

            if not np.all(np.isfinite(result)):
                record["overall_status"] = "failed"
                record["failure_stage"] = "result_finite_check"
                record["failure_reason"] = "nonfinite_exact_age_result"
                return record

            record["overall_status"] = "success"
            return record

        except Exception as exc:
            record["overall_status"] = "failed"
            record["failure_stage"] = "exact_age_result_exception"
            record["failure_reason"] = type(exc).__name__
            record["exception_message"] = str(exc)
            record["traceback"] = traceback.format_exc()
            return record

    # Stage 3: mass interpolation in second isochrone.
    try:
        s2 = interpolator.isoInterp(
            mass,
            a2,
            grid.isochrones,
            filters,
        )
        record["isoInterp_2_returned_none"] = s2 is None
        record["isoInterp_2_result_length"] = (
            len(s2)
            if s2 is not None
            else 0
        )

        if s2 is None:
            record["overall_status"] = "failed"
            record["failure_stage"] = "mass_interpolation_iso2"
            record["failure_reason"] = classify_mass_failure(
                record,
                "iso2",
            )
            return record

    except Exception as exc:
        record["overall_status"] = "failed"
        record["failure_stage"] = "mass_interpolation_iso2_exception"
        record["failure_reason"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        return record

    # Stage 4: age interpolation.
    try:
        result = interpolator.ageInterp(
            age_myr,
            s1,
            a1,
            s2,
            a2,
            grid.log_ages,
        )

        result = np.asarray(
            result,
            dtype=float,
        )

        record["final_result_length"] = len(result)
        record["final_result_all_finite"] = bool(
            np.all(np.isfinite(result))
        )

        if len(result) != 3 + len(filters):
            record["overall_status"] = "failed"
            record["failure_stage"] = "result_shape"
            record["failure_reason"] = "unexpected_result_length"
            return record

        if not np.all(np.isfinite(result)):
            record["overall_status"] = "failed"
            record["failure_stage"] = "result_finite_check"
            record["failure_reason"] = "nonfinite_age_interpolation_result"
            return record

    except Exception as exc:
        record["overall_status"] = "failed"
        record["failure_stage"] = "age_interpolation_exception"
        record["failure_reason"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc()
        return record

    # Compare with public interpolate().
    try:
        public_result = interpolator.interpolate(
            age_myr,
            mass,
            grid.isochrones,
            grid.log_ages,
            filters,
        )

        record["public_interpolate_returned_none"] = (
            public_result is None
        )

        if public_result is not None:
            public_array = np.asarray(
                public_result,
                dtype=float,
            )
            record["public_result_length"] = len(public_array)
            record["public_result_all_finite"] = bool(
                np.all(np.isfinite(public_array))
            )

    except Exception as exc:
        record["public_interpolate_exception_type"] = type(exc).__name__
        record["public_interpolate_exception"] = str(exc)

    record["overall_status"] = "success"
    return record


def classify_mass_failure(
    record: dict[str, Any],
    prefix: str,
) -> str:
    """Classify a None return from isoInterp using inspection metadata."""
    if record.get(f"{prefix}_mass_outside_grid", False):
        mass = float(record["mass"])
        mass_min = record.get(f"{prefix}_mass_min", np.nan)
        mass_max = record.get(f"{prefix}_mass_max", np.nan)

        if mass < mass_min:
            return "mass_below_isochrone_minimum"

        if mass > mass_max:
            return "mass_above_isochrone_maximum"

        return "mass_outside_isochrone_range"

    if record.get(f"{prefix}_inspection_status") == "missing_columns":
        return "missing_required_column"

    if record.get(f"{prefix}_nonfinite_cells", ""):
        return "nonfinite_required_isochrone_value"

    if (
        record.get(f"{prefix}_neighbor_index") is None
        and not record.get(f"{prefix}_exact_mass_match", False)
    ):
        return "no_adjacent_mass_point"

    if record.get(f"{prefix}_zero_mass_denominator", False):
        return "duplicate_mass_zero_width_interval"

    return "isoInterp_returned_None_unclassified"


# %% [markdown]
# ## Scan requested snapshots

# %%
all_system_records = []
failure_records = []

for snapshot_time_myr in SNAPSHOT_TIMES_MYR:
    print("=" * 80)
    print(f"Snapshot time: {snapshot_time_myr:.1f} Myr")

    table = load_cluster_table(
        SIMULATION_PATH,
        snapshot_time_myr,
    )

    masses = np.asarray(
        table["mass"],
        dtype=float,
    )
    ages = np.asarray(
        table["age"],
        dtype=float,
    )

    grid_min_age = float(
        ISO_GRID.ages_myr.min()
    )
    grid_max_age = float(
        ISO_GRID.ages_myr.max()
    )

    snapshot_failures = 0

    for system_index, (mass, raw_age) in enumerate(
        zip(masses, ages)
    ):
        basic = {
            "snapshot_time_myr": float(snapshot_time_myr),
            "system_index": int(system_index),
            "raw_age_myr": float(raw_age),
            "mass": float(mass),
        }

        if not np.isfinite(mass):
            record = {
                **basic,
                "overall_status": "failed",
                "failure_stage": "input_validation",
                "failure_reason": "nonfinite_mass",
            }
            all_system_records.append(record)
            failure_records.append(record)
            snapshot_failures += 1
            continue

        if not np.isfinite(raw_age):
            record = {
                **basic,
                "overall_status": "failed",
                "failure_stage": "input_validation",
                "failure_reason": "nonfinite_age",
            }
            all_system_records.append(record)
            failure_records.append(record)
            snapshot_failures += 1
            continue

        used_age = float(raw_age)
        clipped = False

        if used_age < grid_min_age:
            if CLIP_YOUNG_STARS_TO_GRID_MINIMUM:
                used_age = grid_min_age
                clipped = True
            else:
                record = {
                    **basic,
                    "age_used_myr": used_age,
                    "age_clipped_to_grid": False,
                    "overall_status": "failed",
                    "failure_stage": "input_validation",
                    "failure_reason": "age_below_grid",
                }
                all_system_records.append(record)
                failure_records.append(record)
                snapshot_failures += 1
                continue

        if used_age > grid_max_age:
            record = {
                **basic,
                "age_used_myr": used_age,
                "age_clipped_to_grid": clipped,
                "overall_status": "failed",
                "failure_stage": "input_validation",
                "failure_reason": "age_above_grid",
            }
            all_system_records.append(record)
            failure_records.append(record)
            snapshot_failures += 1
            continue

        diagnostic = diagnose_interpolation(
            used_age,
            float(mass),
            ISO_GRID,
            FILTER_KEYS,
        )

        record = {
            **basic,
            "age_used_myr": used_age,
            "age_clipped_to_grid": clipped,
            **diagnostic,
        }

        all_system_records.append(record)

        if diagnostic["overall_status"] != "success":
            failure_records.append(record)
            snapshot_failures += 1

    print(
        f"  Diagnosed {len(table)} systems; "
        f"found {snapshot_failures} failures."
    )


# %% [markdown]
# ## Summaries and outputs

# %%
df_all_systems = pd.DataFrame(
    all_system_records
)
df_failures = pd.DataFrame(
    failure_records
)

df_all_systems.to_csv(
    OUTPUT_DIR / "all_system_interpolation_diagnostics.csv",
    index=False,
)

df_failures.to_csv(
    OUTPUT_DIR / "failed_system_interpolation_diagnostics.csv",
    index=False,
)

print("\nFailure summary by snapshot and reason:")

if df_failures.empty:
    print("No failures found.")
else:
    failure_summary = (
        df_failures
        .groupby(
            [
                "snapshot_time_myr",
                "failure_stage",
                "failure_reason",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            [
                "snapshot_time_myr",
                "count",
            ],
            ascending=[True, False],
        )
    )

    display(failure_summary)

    failure_summary.to_csv(
        OUTPUT_DIR / "interpolation_failure_summary.csv",
        index=False,
    )

    preferred_columns = [
        "snapshot_time_myr",
        "system_index",
        "raw_age_myr",
        "age_used_myr",
        "mass",
        "failure_stage",
        "failure_reason",
        "age_index_1",
        "age_index_2",
        "age_grid_1_myr",
        "age_grid_2_myr",
        "iso1_mass_min",
        "iso1_mass_max",
        "iso1_nearest_mass",
        "iso1_neighbor_mass",
        "iso1_mass_denominator",
        "iso1_nonfinite_cells",
        "iso2_mass_min",
        "iso2_mass_max",
        "iso2_nearest_mass",
        "iso2_neighbor_mass",
        "iso2_mass_denominator",
        "iso2_nonfinite_cells",
        "exception_message",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in df_failures.columns
    ]

    print("\nDetailed failed systems:")
    display(
        df_failures[existing_columns]
        .sort_values(
            [
                "snapshot_time_myr",
                "system_index",
            ]
        )
    )

print(f"\nDiagnostics saved in: {OUTPUT_DIR.resolve()}")
print("Files:")
print("  all_system_interpolation_diagnostics.csv")
print("  failed_system_interpolation_diagnostics.csv")
print("  interpolation_failure_summary.csv")


# %% [markdown]
# ## Optional: inspect neighboring mass-grid rows around each failure
#
# This table prints a small window of isochrone rows around the nearest
# mass point for every failed system. It is useful for identifying model
# transition points, duplicate masses, discontinuities, or non-finite
# photometry.

# %%
def collect_neighbor_rows(
    failure_row: pd.Series,
    grid: IsoGrid,
    half_window: int = 2,
) -> pd.DataFrame:
    output_rows = []

    for iso_number in [1, 2]:
        age_idx_value = failure_row.get(
            f"age_index_{iso_number}",
            np.nan,
        )

        nearest_idx_value = failure_row.get(
            f"iso{iso_number}_nearest_index",
            np.nan,
        )

        if (
            not np.isfinite(age_idx_value)
            or not np.isfinite(nearest_idx_value)
        ):
            continue

        age_idx = int(age_idx_value)
        nearest_idx = int(nearest_idx_value)

        points = grid.isochrones[age_idx].points

        start = max(
            0,
            nearest_idx - half_window,
        )
        stop = min(
            len(points),
            nearest_idx + half_window + 1,
        )

        for row_idx in range(start, stop):
            row = {
                "snapshot_time_myr": failure_row[
                    "snapshot_time_myr"
                ],
                "system_index": failure_row[
                    "system_index"
                ],
                "requested_mass": failure_row["mass"],
                "requested_age_myr": failure_row[
                    "age_used_myr"
                ],
                "iso_number": iso_number,
                "age_index": age_idx,
                "isochrone_age_myr": float(
                    ISO_GRID.ages_myr[age_idx]
                ),
                "row_index": row_idx,
                "is_nearest_row": row_idx == nearest_idx,
                "mass": float(points[row_idx]["mass"]),
                "L": float(points[row_idx]["L"]),
                "Teff": float(points[row_idx]["Teff"]),
                "logg": float(points[row_idx]["logg"]),
            }

            for filter_name, filter_key in zip(
                FILTER_OBSMODES,
                FILTER_KEYS,
            ):
                row[f"mag_{filter_name}"] = float(
                    points[row_idx][filter_key]
                )

            output_rows.append(row)

    return pd.DataFrame(output_rows)


if not df_failures.empty:
    neighbor_tables = [
        collect_neighbor_rows(row, ISO_GRID)
        for _, row in df_failures.iterrows()
    ]

    neighbor_tables = [
        table
        for table in neighbor_tables
        if not table.empty
    ]

    if neighbor_tables:
        df_neighbor_rows = pd.concat(
            neighbor_tables,
            ignore_index=True,
        )

        display(df_neighbor_rows)

        df_neighbor_rows.to_csv(
            OUTPUT_DIR / "failed_system_neighboring_isochrone_rows.csv",
            index=False,
        )

        print(
            "  failed_system_neighboring_isochrone_rows.csv"
        )
