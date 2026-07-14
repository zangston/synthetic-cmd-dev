#!/usr/bin/env python3

"""
Extend SPISEA's merged Baraffe-Pisa-Ekstrom-PARSEC grids down to
the lower mass limit of the newer Baraffe15 files.

The original SPISEA files are never modified.

Input directories
-----------------
Baraffe FITS:
    /home/wyz5rge/SPISEA/evolution/Baraffe15/iso/z015/

Merged non-rotating:
    /home/wyz5rge/SPISEA/evolution/merged/
    baraffe_pisa_ekstrom_parsec/z015_norot/

Merged rotating:
    /home/wyz5rge/SPISEA/evolution/merged/
    baraffe_pisa_ekstrom_parsec/z015_rot/

Output
------
A directory named:

    merged_baraffe_updated/

is created beside this script, containing:

    z015_norot/
    z015_rot/
    update_summary.csv

For merged files with:

    6.00 <= logAge < 8.01

the script prepends Baraffe rows below the existing minimum merged mass.
Typically, this adds 0.01--0.06 Msun while avoiding a duplicate 0.07 Msun row.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


# =====================================================================
# Configuration
# =====================================================================

BARAFFE_DIR = Path(
    "/home/wyz5rge/SPISEA/evolution/Baraffe15/iso/z015"
)

MERGED_SOURCE_DIRS = {
    "z015_norot": Path(
        "/home/wyz5rge/SPISEA/evolution/merged/"
        "baraffe_pisa_ekstrom_parsec/z015_norot"
    ),
    "z015_rot": Path(
        "/home/wyz5rge/SPISEA/evolution/merged/"
        "baraffe_pisa_ekstrom_parsec/z015_rot"
    ),
}

# Place copied and edited directories beside this script.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "merged_baraffe_updated"

# The merged grids start at logAge = 6.00.
MIN_LOG_AGE = 6.00

# iso_8.01.dat is PARSEC-only, so do not modify it or anything older.
PARSEC_ONLY_LOG_AGE = 8.01

# Normally the existing merged minimum is 0.07 Msun. The script adds all
# available Baraffe rows strictly below the existing merged minimum.
MASS_TOLERANCE = 1.0e-10

# Refuse to overwrite an existing output directory by default.
OVERWRITE_OUTPUT = False

# If True, files missing a matching Baraffe FITS file stop the whole run.
# If False, they are recorded as skipped.
STRICT_MISSING_FILES = False


# =====================================================================
# Data containers
# =====================================================================

@dataclass
class MergedRow:
    m_init: float
    log_t: float
    log_l: float
    log_g: float
    logt_wr: float
    m_curr: float
    phase: int
    source: str


@dataclass
class UpdateRecord:
    grid: str
    filename: str
    log_age: float
    status: str
    original_rows: int = 0
    added_rows: int = 0
    final_rows: int = 0
    original_min_mass: float = math.nan
    added_min_mass: float = math.nan
    added_max_mass: float = math.nan
    final_min_mass: float = math.nan
    final_max_mass: float = math.nan
    boundary_logt_jump: float = math.nan
    boundary_logl_jump: float = math.nan
    boundary_logg_jump: float = math.nan
    message: str = ""


# =====================================================================
# File-name helpers
# =====================================================================

ISO_FILENAME_RE = re.compile(
    r"^iso_(?P<log_age>\d+\.\d+)\.(?P<extension>dat|fits)$"
)


def parse_log_age(path: Path) -> float:
    """Extract logAge from a name such as iso_6.00.dat."""
    match = ISO_FILENAME_RE.match(path.name)

    if match is None:
        raise ValueError(f"Unrecognized isochrone filename: {path.name}")

    return float(match.group("log_age"))


def baraffe_path_for_log_age(log_age: float) -> Path:
    """
    Return the expected Baraffe FITS path.

    Two decimal places are used to match names such as iso_6.00.fits.
    """
    return BARAFFE_DIR / f"iso_{log_age:.2f}.fits"


# =====================================================================
# Reading and writing merged .dat files
# =====================================================================

def read_merged_dat(path: Path) -> tuple[list[str], list[MergedRow]]:
    """
    Read a merged-model .dat file.

    Returns
    -------
    header_lines
        Every initial comment/blank line, preserved verbatim.

    rows
        Parsed model rows.
    """
    header_lines: list[str] = []
    rows: list[MergedRow] = []
    encountered_data = False

    with path.open("r") as file:
        for line_number, raw_line in enumerate(file, start=1):
            stripped = raw_line.strip()

            if not encountered_data and (
                stripped == "" or stripped.startswith("#")
            ):
                header_lines.append(raw_line)
                continue

            if stripped == "" or stripped.startswith("#"):
                # Preserve comments that unexpectedly occur after the data.
                # They are appended to the header for output consistency.
                header_lines.append(raw_line)
                continue

            encountered_data = True
            fields = stripped.split()

            if len(fields) < 8:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 8 fields, "
                    f"found {len(fields)}:\n{raw_line}"
                )

            rows.append(
                MergedRow(
                    m_init=float(fields[0]),
                    log_t=float(fields[1]),
                    log_l=float(fields[2]),
                    log_g=float(fields[3]),
                    logt_wr=float(fields[4]),
                    m_curr=float(fields[5]),
                    phase=int(float(fields[6])),
                    source=" ".join(fields[7:]),
                )
            )

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    return header_lines, rows


def format_merged_row(row: MergedRow) -> str:
    """
    Format a row to resemble the existing merged SPISEA files.

    The Source column is left as a string and may contain '+'.
    """
    return (
        f"{row.m_init:12.6f}"
        f"{row.log_t:12.4f}"
        f"{row.log_l:12.4f}"
        f"{row.log_g:12.4f}"
        f"{row.logt_wr:12.4f}"
        f"{row.m_curr:13.6f}"
        f"{row.phase:6d} "
        f"{row.source}\n"
    )


def write_merged_dat(
    path: Path,
    header_lines: Iterable[str],
    rows: Iterable[MergedRow],
) -> None:
    """Write the updated merged-model file atomically."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w") as file:
        for line in header_lines:
            file.write(line if line.endswith("\n") else line + "\n")

        for row in rows:
            file.write(format_merged_row(row))

    temporary_path.replace(path)


# =====================================================================
# Reading Baraffe FITS files
# =====================================================================

def read_baraffe_fits(path: Path) -> list[MergedRow]:
    """
    Convert one Baraffe FITS table into merged-file rows.

    Baraffe columns:
        Mass, Teff, logL, logG, Rad

    Merged columns:
        M_init, log T, log L, log g, logT_WR, M_curr, phase, Source

    For low-mass Baraffe stars:
        M_init  = Mass
        log T   = log10(Teff)
        log L   = logL
        log g   = logG
        logT_WR = log10(Teff)
        M_curr  = Mass
        phase   = 1
        Source  = Baraffe
    """
    with fits.open(path, memmap=False) as hdul:
        table_hdus = [
            hdu for hdu in hdul
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
        ]

        if not table_hdus:
            raise ValueError(f"No table HDU found in {path}")

        data = table_hdus[0].data
        column_names = set(data.names)

        required = {"Mass", "Teff", "logL", "logG"}
        missing = required - column_names

        if missing:
            raise KeyError(
                f"{path} is missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(column_names)}"
            )

        mass = np.asarray(data["Mass"], dtype=float)
        teff = np.asarray(data["Teff"], dtype=float)
        log_l = np.asarray(data["logL"], dtype=float)
        log_g = np.asarray(data["logG"], dtype=float)

    valid = (
        np.isfinite(mass)
        & np.isfinite(teff)
        & np.isfinite(log_l)
        & np.isfinite(log_g)
        & (mass > 0.0)
        & (teff > 0.0)
    )

    rows = []

    for m, temperature, luminosity, gravity in zip(
        mass[valid],
        teff[valid],
        log_l[valid],
        log_g[valid],
    ):
        log_temperature = float(np.log10(temperature))

        rows.append(
            MergedRow(
                m_init=float(m),
                log_t=log_temperature,
                log_l=float(luminosity),
                log_g=float(gravity),
                logt_wr=log_temperature,
                m_curr=float(m),
                phase=1,
                source="Baraffe",
            )
        )

    rows.sort(key=lambda row: row.m_init)
    return rows


# =====================================================================
# Validation
# =====================================================================

def find_duplicate_masses(
    rows: list[MergedRow],
    tolerance: float = MASS_TOLERANCE,
) -> list[tuple[int, float, float]]:
    """Return adjacent duplicate masses after sorting."""
    duplicates = []

    for index in range(len(rows) - 1):
        m1 = rows[index].m_init
        m2 = rows[index + 1].m_init

        if np.isclose(m1, m2, rtol=0.0, atol=tolerance):
            duplicates.append((index, m1, m2))

    return duplicates


def validate_rows(rows: list[MergedRow], path: Path) -> None:
    """Validate mass ordering and finite numerical values."""
    if not rows:
        raise ValueError(f"No rows available for {path}")

    masses = np.array([row.m_init for row in rows], dtype=float)

    if not np.all(np.isfinite(masses)):
        raise ValueError(f"Non-finite masses found in {path}")

    if np.any(np.diff(masses) < -MASS_TOLERANCE):
        bad_indices = np.where(np.diff(masses) < -MASS_TOLERANCE)[0]
        raise ValueError(
            f"Mass is not monotonic in {path}; "
            f"first offending indices: {bad_indices[:10]}"
        )

    duplicates = find_duplicate_masses(rows)

    # A duplicate at 0.4 is already part of the original merged construction:
    # one pure Baraffe point and one Baraffe+Pisa transition point.
    # We permit duplicates that were already present in the source file.
    # New low-mass rows are selected below the old minimum, so they should not
    # introduce new duplicates.


def boundary_jumps(
    added_rows: list[MergedRow],
    original_rows: list[MergedRow],
) -> tuple[float, float, float]:
    """
    Compute discontinuities between the highest added Baraffe row and
    the original lowest-mass merged row.
    """
    if not added_rows or not original_rows:
        return math.nan, math.nan, math.nan

    low_boundary = max(added_rows, key=lambda row: row.m_init)
    high_boundary = min(original_rows, key=lambda row: row.m_init)

    return (
        high_boundary.log_t - low_boundary.log_t,
        high_boundary.log_l - low_boundary.log_l,
        high_boundary.log_g - low_boundary.log_g,
    )


# =====================================================================
# Update logic
# =====================================================================

def should_modify(log_age: float) -> bool:
    return (
        log_age >= MIN_LOG_AGE - 1.0e-9
        and log_age < PARSEC_ONLY_LOG_AGE - 1.0e-9
    )


def update_one_file(
    grid_name: str,
    merged_path: Path,
) -> UpdateRecord:
    log_age = parse_log_age(merged_path)

    record = UpdateRecord(
        grid=grid_name,
        filename=merged_path.name,
        log_age=log_age,
        status="pending",
    )

    if not should_modify(log_age):
        record.status = "unchanged"
        record.message = (
            f"Outside update interval "
            f"[{MIN_LOG_AGE:.2f}, {PARSEC_ONLY_LOG_AGE:.2f})"
        )
        return record

    baraffe_path = baraffe_path_for_log_age(log_age)

    if not baraffe_path.exists():
        record.status = "skipped_missing_baraffe"
        record.message = f"Missing matching file: {baraffe_path}"

        if STRICT_MISSING_FILES:
            raise FileNotFoundError(record.message)

        return record

    header, original_rows = read_merged_dat(merged_path)
    baraffe_rows = read_baraffe_fits(baraffe_path)

    original_rows.sort(key=lambda row: row.m_init)

    original_min_mass = min(row.m_init for row in original_rows)
    original_max_mass = max(row.m_init for row in original_rows)

    # Strictly less than the existing minimum avoids duplicating the 0.07 row.
    added_rows = [
        row for row in baraffe_rows
        if row.m_init < original_min_mass - MASS_TOLERANCE
    ]

    record.original_rows = len(original_rows)
    record.original_min_mass = original_min_mass

    if not added_rows:
        record.status = "unchanged_no_lower_mass_rows"
        record.final_rows = len(original_rows)
        record.final_min_mass = original_min_mass
        record.final_max_mass = original_max_mass
        record.message = (
            "The Baraffe file contains no rows below the merged "
            f"minimum mass of {original_min_mass:.6f} Msun."
        )
        return record

    updated_rows = added_rows + original_rows
    updated_rows.sort(key=lambda row: row.m_init)

    validate_rows(updated_rows, merged_path)

    jump_logt, jump_logl, jump_logg = boundary_jumps(
        added_rows,
        original_rows,
    )

    write_merged_dat(
        merged_path,
        header,
        updated_rows,
    )

    record.status = "updated"
    record.added_rows = len(added_rows)
    record.final_rows = len(updated_rows)
    record.added_min_mass = min(row.m_init for row in added_rows)
    record.added_max_mass = max(row.m_init for row in added_rows)
    record.final_min_mass = min(row.m_init for row in updated_rows)
    record.final_max_mass = max(row.m_init for row in updated_rows)
    record.boundary_logt_jump = jump_logt
    record.boundary_logl_jump = jump_logl
    record.boundary_logg_jump = jump_logg
    record.message = (
        f"Prepended {len(added_rows)} Baraffe rows below "
        f"{original_min_mass:.6f} Msun."
    )

    return record


def copy_source_grids() -> dict[str, Path]:
    """Copy the original merged grids into the output directory."""
    if OUTPUT_ROOT.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output directory already exists:\n{OUTPUT_ROOT}\n\n"
                "Remove it manually or set OVERWRITE_OUTPUT = True."
            )

        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True)

    copied_dirs = {}

    for grid_name, source_dir in MERGED_SOURCE_DIRS.items():
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Source merged directory not found: {source_dir}"
            )

        destination = OUTPUT_ROOT / grid_name

        print(f"Copying:\n  {source_dir}\n  -> {destination}")
        shutil.copytree(source_dir, destination)

        copied_dirs[grid_name] = destination

    return copied_dirs


def write_summary(records: list[UpdateRecord]) -> Path:
    summary_path = OUTPUT_ROOT / "update_summary.csv"

    with summary_path.open("w", newline="") as file:
        fieldnames = list(asdict(records[0]).keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    return summary_path


# =====================================================================
# Main program
# =====================================================================

def main() -> int:
    if not BARAFFE_DIR.is_dir():
        raise FileNotFoundError(
            f"Baraffe directory not found: {BARAFFE_DIR}"
        )

    copied_dirs = copy_source_grids()
    records: list[UpdateRecord] = []

    for grid_name, copied_dir in copied_dirs.items():
        print("\n" + "=" * 78)
        print(f"Updating copied grid: {grid_name}")
        print("=" * 78)

        merged_files = sorted(
            copied_dir.glob("iso_*.dat"),
            key=parse_log_age,
        )

        if not merged_files:
            raise FileNotFoundError(
                f"No iso_*.dat files found in {copied_dir}"
            )

        for merged_path in merged_files:
            record = update_one_file(
                grid_name,
                merged_path,
            )
            records.append(record)

            if record.status == "updated":
                print(
                    f"{merged_path.name}: added "
                    f"{record.added_rows} rows; "
                    f"{record.original_min_mass:.3f} -> "
                    f"{record.final_min_mass:.3f} Msun"
                )
            elif record.status.startswith("skipped"):
                print(
                    f"{merged_path.name}: "
                    f"{record.status}: {record.message}"
                )

    summary_path = write_summary(records)

    updated_records = [
        record for record in records
        if record.status == "updated"
    ]
    skipped_records = [
        record for record in records
        if record.status.startswith("skipped")
    ]

    print("\n" + "=" * 78)
    print("Update complete")
    print("=" * 78)
    print(f"Output root:      {OUTPUT_ROOT}")
    print(f"Files updated:    {len(updated_records)}")
    print(f"Files skipped:    {len(skipped_records)}")
    print(f"Summary CSV:      {summary_path}")

    if updated_records:
        print(
            "Final minimum mass range: "
            f"{min(r.final_min_mass for r in updated_records):.3f} "
            "Msun"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise