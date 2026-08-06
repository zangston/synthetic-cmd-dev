# %% [markdown]
# # Diagnose NBODY6 component-to-system conversion
#
# This diagnostic:
#
# - loads one NBODY6 snapshot;
# - converts it with `converter.to_spicea_table()`;
# - inspects the snapshot and `snapshot.stars`;
# - searches for persistent identifiers and binary/system mapping arrays;
# - prints the source location and implementation of
#   `converter.to_spicea_table()`.
#
# It is intended to determine how persistent NBODY6 `NAME` values should be
# propagated into the SPISEA system table without relying on mass matching.

# %%
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nbody6tools import Reader
from nbody62spisea import converter


# %% [markdown]
# ## Configuration

# %%
SIMULATION_PATH = Path(
    "/standard/Tan_JC/backup_protoclusters/multiples/M3000new/"
    "sigma0p1/fiducial/sfe_ff003/00"
)

DIAGNOSTIC_TIME_MYR = 1.5

# Limit large array previews so notebook output stays manageable.
MAX_PREVIEW_VALUES = 30


# %% [markdown]
# ## Validation and snapshot loading

# %%
if not SIMULATION_PATH.is_dir():
    raise FileNotFoundError(
        f"Simulation directory does not exist: {SIMULATION_PATH}"
    )

simulation_path_string = os.path.abspath(
    str(SIMULATION_PATH)
)

if not simulation_path_string.endswith("/"):
    simulation_path_string += "/"

print("Simulation path:", simulation_path_string)
print("Requested diagnostic time:", DIAGNOSTIC_TIME_MYR, "Myr")

snapshot = Reader.read_snapshot(
    simulation_path_string,
    time=float(DIAGNOSTIC_TIME_MYR),
)

snapshot.to_physical()

converted = converter.to_spicea_table(snapshot)


# %% [markdown]
# ## Basic converted-table information

# %%
print("=" * 80)
print("CONVERTED TABLE")
print("=" * 80)

print("Rows:", len(converted))
print("Columns:", converted.colnames)

display(converted[:10])


# %% [markdown]
# ## Snapshot top-level attributes

# %%
print()
print("=" * 80)
print("SNAPSHOT TOP-LEVEL ATTRIBUTES")
print("=" * 80)

snapshot_public = [
    name
    for name in dir(snapshot)
    if not name.startswith("_")
]

snapshot_attribute_records = []

for name in snapshot_public:
    try:
        value = getattr(snapshot, name)

        if callable(value):
            continue

        try:
            length = len(value)
        except Exception:
            length = None

        snapshot_attribute_records.append(
            {
                "attribute": name,
                "type": type(value).__name__,
                "length": length,
            }
        )

    except Exception as exc:
        snapshot_attribute_records.append(
            {
                "attribute": name,
                "type": "ERROR",
                "length": None,
                "error": str(exc),
            }
        )

df_snapshot_attributes = pd.DataFrame(
    snapshot_attribute_records
)

if not df_snapshot_attributes.empty:
    sort_columns = [
        column
        for column in ["length", "attribute"]
        if column in df_snapshot_attributes.columns
    ]

    df_snapshot_attributes = (
        df_snapshot_attributes
        .sort_values(
            sort_columns,
            na_position="last",
        )
        .reset_index(drop=True)
    )

display(df_snapshot_attributes)


# %% [markdown]
# ## Inspect `snapshot.stars`

# %%
if not hasattr(snapshot, "stars"):
    raise AttributeError(
        "The loaded snapshot does not have a `stars` attribute."
    )

stars = snapshot.stars

print()
print("=" * 80)
print("SNAPSHOT.STARS ATTRIBUTES")
print("=" * 80)

star_public = [
    name
    for name in dir(stars)
    if not name.startswith("_")
]

star_attribute_records = []

for name in star_public:
    try:
        value = getattr(stars, name)

        if callable(value):
            continue

        try:
            array = np.asarray(value)
            length = len(array)
            shape = array.shape
            dtype = str(array.dtype)
        except Exception:
            length = None
            shape = None
            dtype = None

        star_attribute_records.append(
            {
                "attribute": name,
                "type": type(value).__name__,
                "length": length,
                "shape": str(shape),
                "dtype": dtype,
                "error": "",
            }
        )

    except Exception as exc:
        star_attribute_records.append(
            {
                "attribute": name,
                "type": "ERROR",
                "length": None,
                "shape": "",
                "dtype": "",
                "error": str(exc),
            }
        )

df_star_attributes = pd.DataFrame(
    star_attribute_records
)

if not df_star_attributes.empty:
    df_star_attributes = (
        df_star_attributes
        .sort_values(
            ["length", "attribute"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

display(df_star_attributes)


# %% [markdown]
# ## Likely identifier and binary-mapping attributes

# %%
print()
print("=" * 80)
print("LIKELY ID / BINARY-MAPPING ATTRIBUTES")
print("=" * 80)

keywords = (
    "name",
    "id",
    "binary",
    "bin",
    "comp",
    "primary",
    "secondary",
    "pair",
    "index",
    "parent",
    "system",
    "kstar",
    "kw",
    "mass",
)

likely_attributes = [
    name
    for name in star_public
    if any(
        keyword in name.lower()
        for keyword in keywords
    )
]

for name in likely_attributes:
    try:
        value = getattr(stars, name)

        if callable(value):
            continue

        array = np.asarray(value)

        print()
        print(f"stars.{name}")
        print("  Python type:", type(value))
        print("  shape:", array.shape)
        print("  dtype:", array.dtype)

        if array.size > 0:
            flattened = array.reshape(-1)
            print(
                "  first values:",
                flattened[:MAX_PREVIEW_VALUES],
            )
        else:
            print("  array is empty")

    except Exception as exc:
        print(
            f"stars.{name}: "
            f"{type(exc).__name__}: {exc}"
        )


# %% [markdown]
# ## Basic star-count comparison

# %%
print()
print("=" * 80)
print("STAR-COUNT COMPARISON")
print("=" * 80)

snapshot_star_count = None

for candidate in (
    "name",
    "mass",
    "m",
):
    if hasattr(stars, candidate):
        try:
            candidate_array = np.asarray(
                getattr(stars, candidate)
            )
            snapshot_star_count = len(candidate_array)

            print(
                f"Using stars.{candidate} for component count: "
                f"{snapshot_star_count}"
            )
            break

        except Exception:
            pass

print("Converted SPISEA-system rows:", len(converted))

if snapshot_star_count is not None:
    print(
        "Difference, components minus converted systems:",
        snapshot_star_count - len(converted),
    )


# %% [markdown]
# ## Other possible snapshot containers

# %%
print()
print("=" * 80)
print("OTHER SNAPSHOT CONTAINERS")
print("=" * 80)

possible_container_names = (
    "binaries",
    "binary",
    "pairs",
    "multiples",
    "systems",
)

found_container = False

for container_name in possible_container_names:
    if not hasattr(snapshot, container_name):
        continue

    found_container = True
    container = getattr(
        snapshot,
        container_name,
    )

    print()
    print(f"snapshot.{container_name}")
    print("type:", type(container))

    public = [
        name
        for name in dir(container)
        if not name.startswith("_")
    ]

    for name in public:
        try:
            value = getattr(container, name)

            if callable(value):
                continue

            try:
                array = np.asarray(value)
                flattened = array.reshape(-1)

                preview = flattened[
                    :min(
                        MAX_PREVIEW_VALUES,
                        len(flattened),
                    )
                ]

                print(
                    f"  {name:25s} "
                    f"shape={str(array.shape):15s} "
                    f"dtype={str(array.dtype):12s} "
                    f"first={preview}"
                )

            except Exception:
                print(
                    f"  {name:25s} "
                    f"type={type(value).__name__}"
                )

        except Exception as exc:
            print(
                f"  {name:25s} "
                f"ERROR: {type(exc).__name__}: {exc}"
            )

if not found_container:
    print(
        "No top-level binary/pair/multiple/system "
        "containers were found."
    )


# %% [markdown]
# ## Converter module and function locations

# %%
print()
print("=" * 80)
print("CONVERTER SOURCE LOCATIONS")
print("=" * 80)

try:
    print(
        "converter module file:",
        inspect.getfile(converter),
    )
except Exception as exc:
    print(
        "Could not locate converter module:",
        f"{type(exc).__name__}: {exc}",
    )

try:
    print(
        "to_spicea_table function file:",
        inspect.getfile(
            converter.to_spicea_table
        ),
    )
except Exception as exc:
    print(
        "Could not locate to_spicea_table():",
        f"{type(exc).__name__}: {exc}",
    )


# %% [markdown]
# ## Print `converter.to_spicea_table()` source

# %%
print()
print("=" * 80)
print("converter.to_spicea_table SOURCE")
print("=" * 80)

try:
    source = inspect.getsource(
        converter.to_spicea_table
    )
    print(source)

except Exception as exc:
    print(
        "Could not retrieve source with "
        "inspect.getsource():",
        f"{type(exc).__name__}: {exc}",
    )

    try:
        print(
            "Open this file manually:",
            inspect.getfile(converter),
        )
    except Exception:
        pass


# %% [markdown]
# ## Optional direct previews of core arrays

# %%
print()
print("=" * 80)
print("CORE SNAPSHOT ARRAY PREVIEWS")
print("=" * 80)

for attribute_name in (
    "name",
    "mass",
    "m",
    "age",
    "kstar",
    "kw",
):
    if not hasattr(stars, attribute_name):
        continue

    try:
        array = np.asarray(
            getattr(stars, attribute_name)
        )

        print()
        print(f"stars.{attribute_name}")
        print("  shape:", array.shape)
        print("  dtype:", array.dtype)
        print(
            "  first values:",
            array.reshape(-1)[:MAX_PREVIEW_VALUES],
        )

    except Exception as exc:
        print(
            f"stars.{attribute_name}: "
            f"{type(exc).__name__}: {exc}"
        )


# %% [markdown]
# ## Summary

# %%
print()
print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(
    f"Snapshot requested at "
    f"{DIAGNOSTIC_TIME_MYR:g} Myr"
)

if snapshot_star_count is not None:
    print(
        f"Snapshot component count: "
        f"{snapshot_star_count}"
    )

print(
    f"Converted system count: "
    f"{len(converted)}"
)

print(
    "Next step: inspect the printed "
    "`to_spicea_table()` implementation for the exact "
    "indexing, masking, sorting, and binary grouping logic."
)

# %% [markdown]
# ## Inspect unresolved binary pointers and converter implementation

# %%
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nbody6tools import Reader
from nbody62spisea import converter


SIMULATION_PATH = Path(
    "/standard/Tan_JC/backup_protoclusters/multiples/M3000new/"
    "sigma0p1/fiducial/sfe_ff003/00"
)

DIAGNOSTIC_TIME_MYR = 1.5
MAX_PREVIEW = 100


path = os.path.abspath(str(SIMULATION_PATH))
if not path.endswith("/"):
    path += "/"

snapshot = Reader.read_snapshot(
    path,
    time=float(DIAGNOSTIC_TIME_MYR),
)
snapshot.to_physical()

converted = converter.to_spicea_table(snapshot)


print("=" * 80)
print("UNRESOLVED POINTERS")
print("=" * 80)

print("Python type:", type(snapshot.unresolved_pointers))
print("Tuple length:", len(snapshot.unresolved_pointers))

for i, item in enumerate(snapshot.unresolved_pointers):
    print()
    print(f"unresolved_pointers[{i}]")
    print("  Python type:", type(item))

    try:
        arr = np.asarray(item)

        print("  shape:", arr.shape)
        print("  dtype:", arr.dtype)
        print("  size:", arr.size)
        print("  first values:", arr.reshape(-1)[:MAX_PREVIEW])

        if arr.dtype == bool:
            print("  number True:", int(arr.sum()))
            print(
                "  first True indices:",
                np.where(arr.reshape(-1))[0][:MAX_PREVIEW],
            )

    except Exception as exc:
        print(
            "  Could not convert to NumPy array:",
            f"{type(exc).__name__}: {exc}",
        )


print()
print("=" * 80)
print("UNRESOLVED STAR CONTAINERS")
print("=" * 80)

for container_name in (
    "unresolved_stars",
    "bound_stars_unresolved",
    "unbound_stars_unresolved",
):
    container = getattr(snapshot, container_name, None)

    if container is None:
        print(f"{container_name}: not present")
        continue

    print()
    print(container_name)
    print("  type:", type(container))

    for attr in ("name", "mass", "I", "kstar"):
        if not hasattr(container, attr):
            continue

        arr = np.asarray(getattr(container, attr))

        print(
            f"  {attr:8s}: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"first={arr.reshape(-1)[:30]}"
        )


print()
print("=" * 80)
print("ALL PARTICLES")
print("=" * 80)

allparticles = snapshot.allparticles

for attr in ("name", "mass", "I", "kstar"):
    if not hasattr(allparticles, attr):
        continue

    arr = np.asarray(getattr(allparticles, attr))

    print(
        f"{attr:8s}: "
        f"shape={arr.shape}, "
        f"dtype={arr.dtype}, "
        f"first={arr.reshape(-1)[:30]}"
    )


print()
print("=" * 80)
print("BINARY COUNTS FROM CONVERTED TABLE")
print("=" * 80)

is_multiple = np.asarray(converted["isMultiple"], dtype=float)

print("Converted systems:", len(converted))
print(
    "Single systems:",
    int(np.sum(is_multiple == 0)),
)
print(
    "Multiple systems:",
    int(np.sum(is_multiple != 0)),
)

companion_counts = []

for value in converted["compMass"]:
    try:
        companion_counts.append(len(value))
    except TypeError:
        companion_counts.append(0)

companion_counts = np.asarray(companion_counts, dtype=int)

print(
    "Total companion components represented:",
    int(companion_counts.sum()),
)
print(
    "Rows with at least one companion:",
    int(np.sum(companion_counts > 0)),
)
print(
    "Maximum companions in one system:",
    int(companion_counts.max()),
)


print()
print("=" * 80)
print("CONVERTER SOURCE FILE")
print("=" * 80)

converter_file = Path(inspect.getfile(converter))
print(converter_file)


print()
print("=" * 80)
print("converter.to_spicea_table SOURCE")
print("=" * 80)

try:
    print(inspect.getsource(converter.to_spicea_table))
except Exception as exc:
    print(
        "inspect.getsource failed:",
        f"{type(exc).__name__}: {exc}",
    )

    print()
    print("Reading converter.py directly instead:")

    text = converter_file.read_text()

    function_start = text.find("def to_spicea_table")

    if function_start < 0:
        print("Could not find `def to_spicea_table` in converter.py")
    else:
        next_function = text.find(
            "\ndef ",
            function_start + 1,
        )

        if next_function < 0:
            next_function = len(text)

        print(text[function_start:next_function])

# %% [markdown]
# ## Characterize unresolved_pointers and inspect converter source

# %%
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from nbody62spisea import converter


ptr0 = np.asarray(snapshot.unresolved_pointers[0], dtype=int)
ptr1 = np.asarray(snapshot.unresolved_pointers[1], dtype=int)

n_components = len(snapshot.stars)


print("=" * 80)
print("POINTER COMPARISON")
print("=" * 80)

print("Component count:", n_components)
print("ptr0 length:", len(ptr0))
print("ptr1 length:", len(ptr1))
print("Arrays exactly equal:", np.array_equal(ptr0, ptr1))

print()
print("ptr0 minimum:", ptr0.min())
print("ptr0 maximum:", ptr0.max())
print("ptr0 unique values:", len(np.unique(ptr0)))
print("ptr0 duplicated entries:", len(ptr0) - len(np.unique(ptr0)))

print()
print("ptr1 minimum:", ptr1.min())
print("ptr1 maximum:", ptr1.max())
print("ptr1 unique values:", len(np.unique(ptr1)))
print("ptr1 duplicated entries:", len(ptr1) - len(np.unique(ptr1)))


print()
print("=" * 80)
print("MISSING AND OUT-OF-RANGE INDICES")
print("=" * 80)

# Test both common indexing conventions.
expected_zero_based = np.arange(n_components)
expected_one_based = np.arange(1, n_components + 1)

missing_zero_based = np.setdiff1d(
    expected_zero_based,
    np.unique(ptr0),
)

missing_one_based = np.setdiff1d(
    expected_one_based,
    np.unique(ptr0),
)

outside_zero_based = ptr0[
    (ptr0 < 0) | (ptr0 >= n_components)
]

outside_one_based = ptr0[
    (ptr0 < 1) | (ptr0 > n_components)
]

print("Assuming zero-based component indices:")
print("  Missing indices:", missing_zero_based)
print("  Out-of-range values:", np.unique(outside_zero_based))

print()
print("Assuming one-based component indices:")
print("  Missing indices:", missing_one_based)
print("  Out-of-range values:", np.unique(outside_one_based))


print()
print("=" * 80)
print("ORDERING TESTS")
print("=" * 80)

print(
    "ptr0 is strictly increasing:",
    bool(np.all(np.diff(ptr0) > 0)),
)

print(
    "ptr0 is nondecreasing:",
    bool(np.all(np.diff(ptr0) >= 0)),
)

print(
    "Sorted ptr0 first 50:",
    np.sort(ptr0)[:50],
)

print(
    "Sorted ptr0 last 50:",
    np.sort(ptr0)[-50:],
)


print()
print("=" * 80)
print("POINTER-SELECTED STAR VALUES")
print("=" * 80)

# Only index the component arrays if the pointers are valid zero-based indices.
if np.all((ptr0 >= 0) & (ptr0 < n_components)):
    selected_names = np.asarray(snapshot.stars.name)[ptr0]
    selected_masses = np.asarray(snapshot.stars.mass)[ptr0]
    selected_kstar = np.asarray(snapshot.stars.kstar)[ptr0]

    print("Selected NAME values:")
    print(selected_names[:100])

    print()
    print("Selected masses:")
    print(selected_masses[:100])

    print()
    print("Selected kstar values:")
    print(selected_kstar[:100])
else:
    print(
        "Pointers are not all valid zero-based component indices; "
        "skipping direct indexing."
    )


print()
print("=" * 80)
print("CONVERTER SOURCE")
print("=" * 80)

converter_path = Path(inspect.getfile(converter))
print("Converter file:", converter_path)

try:
    print(inspect.getsource(converter.to_spicea_table))
except Exception as exc:
    print(
        "inspect.getsource failed:",
        f"{type(exc).__name__}: {exc}",
    )

    source_text = converter_path.read_text()

    start = source_text.find("def to_spicea_table")

    if start == -1:
        print("Could not locate `def to_spicea_table`.")
    else:
        next_def = source_text.find("\ndef ", start + 1)

        if next_def == -1:
            next_def = len(source_text)

        print(source_text[start:next_def])

# %% [markdown]
# ## Interpret unresolved_pointers
#
# Test whether ptr1 is a one-based selector into snapshot.stars and whether
# ptr0 corresponds to NAME, I, or another persistent/system identifier.

# %%
import numpy as np
import pandas as pd


ptr0 = np.asarray(snapshot.unresolved_pointers[0], dtype=int)
ptr1 = np.asarray(snapshot.unresolved_pointers[1], dtype=int)

n_stars = len(snapshot.stars)

# ptr1 appears to use Fortran/NBODY6 one-based indexing.
star_idx = ptr1 - 1

if np.any(star_idx < 0) or np.any(star_idx >= n_stars):
    raise RuntimeError(
        "ptr1 cannot safely be interpreted as one-based indices into "
        "snapshot.stars."
    )

star_names = np.asarray(snapshot.stars.name, dtype=int)
star_I = np.asarray(snapshot.stars.I, dtype=int)
star_masses = np.asarray(snapshot.stars.mass, dtype=float)
star_kstar = np.asarray(snapshot.stars.kstar, dtype=int)

selected_names = star_names[star_idx]
selected_I = star_I[star_idx]
selected_masses = star_masses[star_idx]
selected_kstar = star_kstar[star_idx]


print("=" * 80)
print("MISSING PTR1 INDEX")
print("=" * 80)

expected_one_based = np.arange(1, n_stars + 1)
missing_ptr1 = np.setdiff1d(expected_one_based, ptr1)

print("Missing one-based snapshot.stars index:", missing_ptr1)

if len(missing_ptr1) == 1:
    omitted_idx = int(missing_ptr1[0] - 1)

    print("Omitted zero-based index:", omitted_idx)
    print("Omitted NAME:", star_names[omitted_idx])
    print("Omitted I:", star_I[omitted_idx])
    print("Omitted mass:", star_masses[omitted_idx])
    print("Omitted kstar:", star_kstar[omitted_idx])


print()
print("=" * 80)
print("DOES PTR0 MATCH A STAR ATTRIBUTE?")
print("=" * 80)

comparisons = {
    "ptr0 == selected stars.name": ptr0 == selected_names,
    "ptr0 == selected stars.I": ptr0 == selected_I,
    "ptr0 == ptr1": ptr0 == ptr1,
}

for label, match in comparisons.items():
    print(
        f"{label:32s}: "
        f"{match.sum():5d}/{len(match)} "
        f"({match.mean():.6f})"
    )


print()
print("=" * 80)
print("FIRST 100 POINTER ASSOCIATIONS")
print("=" * 80)

df_pointer_map = pd.DataFrame(
    {
        "pointer_row": np.arange(len(ptr0)),
        "ptr0": ptr0,
        "ptr1_one_based": ptr1,
        "star_index_zero_based": star_idx,
        "selected_name": selected_names,
        "selected_I": selected_I,
        "selected_mass": selected_masses,
        "selected_kstar": selected_kstar,
        "ptr0_equals_name": ptr0 == selected_names,
        "ptr0_equals_I": ptr0 == selected_I,
    }
)

display(df_pointer_map.head(100))


print()
print("=" * 80)
print("PTR0 DUPLICATE-GROUP SUMMARY")
print("=" * 80)

ptr0_groups = (
    df_pointer_map.groupby("ptr0", as_index=False)
    .agg(
        n_components=("ptr0", "size"),
        component_names=(
            "selected_name",
            lambda values: list(values),
        ),
        component_indices=(
            "star_index_zero_based",
            lambda values: list(values),
        ),
        component_masses=(
            "selected_mass",
            lambda values: list(values),
        ),
        component_kstar=(
            "selected_kstar",
            lambda values: list(values),
        ),
    )
    .sort_values(
        ["n_components", "ptr0"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print("Number of ptr0 groups:", len(ptr0_groups))
print(
    "Group-size counts:"
)
display(
    ptr0_groups["n_components"]
    .value_counts()
    .sort_index()
    .rename_axis("components_per_ptr0")
    .reset_index(name="n_groups")
)

print()
print("Largest groups:")
display(ptr0_groups.head(50))


print()
print("=" * 80)
print("COMPARE GROUP COUNTS TO CONVERTED SYSTEMS")
print("=" * 80)

print("Snapshot star components:", len(snapshot.stars))
print("Pointer rows:", len(ptr0))
print("Unique ptr0 groups:", ptr0_groups["ptr0"].nunique())
print("Converted systems:", len(converted))
print(
    "Converted multiple systems:",
    int(np.count_nonzero(
        np.asarray(converted["isMultiple"], dtype=float)
    )),
)

companion_counts = np.array(
    [
        len(value)
        for value in converted["compMass"]
    ],
    dtype=int,
)

print("Converted companion components:", companion_counts.sum())
print(
    "Converted systems with companions:",
    np.count_nonzero(companion_counts > 0),
)
print(
    "Maximum companions in one converted system:",
    companion_counts.max(),
)

# %% [markdown]
# ## Map unresolved_pointers to unresolved-system catalogs
#
# This checks whether ptr0 identifies entries in snapshot.unresolved_stars
# or snapshot.bound_stars_unresolved, and compares those catalogs with the
# output of converter.to_spicea_table().

# %%
import numpy as np
import pandas as pd


ptr0 = np.asarray(snapshot.unresolved_pointers[0], dtype=int)
ptr1 = np.asarray(snapshot.unresolved_pointers[1], dtype=int)
component_idx = ptr1 - 1

components = snapshot.stars
all_unresolved = snapshot.unresolved_stars
bound_unresolved = snapshot.bound_stars_unresolved


def particle_catalog(container, label):
    """Convert available particle identifiers and masses to a DataFrame."""
    columns = {}

    for attr in ("I", "name", "mass", "kstar"):
        if hasattr(container, attr):
            value = np.asarray(getattr(container, attr))

            if value.ndim == 1 and len(value) == len(container):
                columns[attr] = value

    df = pd.DataFrame(columns)
    df.insert(0, "catalog_index", np.arange(len(container)))
    df.insert(0, "catalog", label)

    return df


df_components = particle_catalog(
    components,
    "snapshot.stars",
)

df_all_unresolved = particle_catalog(
    all_unresolved,
    "snapshot.unresolved_stars",
)

df_bound_unresolved = particle_catalog(
    bound_unresolved,
    "snapshot.bound_stars_unresolved",
)


print("=" * 80)
print("CATALOG SIZES")
print("=" * 80)

print("snapshot.stars:", len(df_components))
print("snapshot.unresolved_stars:", len(df_all_unresolved))
print(
    "snapshot.bound_stars_unresolved:",
    len(df_bound_unresolved),
)
print("converted SPISEA systems:", len(converted))
print("unique ptr0 values:", len(np.unique(ptr0)))


print()
print("=" * 80)
print("PTR0 MEMBERSHIP TESTS")
print("=" * 80)

unique_ptr0 = np.unique(ptr0)

for label, df in [
    ("all unresolved", df_all_unresolved),
    ("bound unresolved", df_bound_unresolved),
]:
    print()
    print(label)

    for identifier in ("I", "name"):
        if identifier not in df.columns:
            continue

        values = np.asarray(df[identifier], dtype=int)

        n_ptr_in_catalog = np.isin(
            unique_ptr0,
            values,
        ).sum()

        n_catalog_in_ptr = np.isin(
            values,
            unique_ptr0,
        ).sum()

        print(
            f"  ptr0 values found in {identifier}: "
            f"{n_ptr_in_catalog}/{len(unique_ptr0)}"
        )
        print(
            f"  catalog {identifier} values found in ptr0: "
            f"{n_catalog_in_ptr}/{len(values)}"
        )


print()
print("=" * 80)
print("POINTER-DERIVED SYSTEM TABLE")
print("=" * 80)

component_names = np.asarray(components.name, dtype=int)
component_I = np.asarray(components.I, dtype=int)
component_mass = np.asarray(components.mass, dtype=float)
component_kstar = np.asarray(components.kstar, dtype=int)

df_pointer_components = pd.DataFrame(
    {
        "ptr0": ptr0,
        "ptr1_one_based": ptr1,
        "component_index": component_idx,
        "component_name": component_names[component_idx],
        "component_I": component_I[component_idx],
        "component_mass": component_mass[component_idx],
        "component_kstar": component_kstar[component_idx],
    }
)

df_pointer_systems = (
    df_pointer_components
    .sort_values(
        ["ptr0", "component_mass"],
        ascending=[True, False],
    )
    .groupby("ptr0", as_index=False)
    .agg(
        n_components=("component_mass", "size"),
        primary_mass=("component_mass", "first"),
        system_mass=("component_mass", "sum"),
        component_masses=(
            "component_mass",
            lambda values: list(values),
        ),
        component_names=(
            "component_name",
            lambda values: list(values),
        ),
        component_I=(
            "component_I",
            lambda values: list(values),
        ),
    )
)

print("Pointer-derived systems:", len(df_pointer_systems))
print(
    "Pointer-derived binary systems:",
    (df_pointer_systems["n_components"] > 1).sum(),
)

display(df_pointer_systems.head(20))


print()
print("=" * 80)
print("MATCH PTR0 TO UNRESOLVED CATALOG IDENTIFIERS")
print("=" * 80)


def attach_unresolved_catalog(pointer_systems, unresolved_df, label):
    results = []

    for identifier in ("I", "name"):
        if identifier not in unresolved_df.columns:
            continue

        right = unresolved_df.copy().rename(
            columns={
                "catalog_index": f"{label}_catalog_index",
                "mass": f"{label}_mass",
                "kstar": f"{label}_kstar",
            }
        )

        merged = pointer_systems.merge(
            right,
            left_on="ptr0",
            right_on=identifier,
            how="left",
            suffixes=("", f"_{label}"),
        )

        match_count = merged[
            f"{label}_catalog_index"
        ].notna().sum()

        results.append(
            {
                "catalog": label,
                "matched_using": identifier,
                "matched_systems": match_count,
                "pointer_systems": len(pointer_systems),
                "match_fraction": (
                    match_count / len(pointer_systems)
                ),
            }
        )

    return results


identifier_results = []

identifier_results.extend(
    attach_unresolved_catalog(
        df_pointer_systems,
        df_all_unresolved,
        "all_unresolved",
    )
)

identifier_results.extend(
    attach_unresolved_catalog(
        df_pointer_systems,
        df_bound_unresolved,
        "bound_unresolved",
    )
)

df_identifier_results = pd.DataFrame(
    identifier_results
)

display(df_identifier_results)


print()
print("=" * 80)
print("MASS-DISTRIBUTION COMPARISON")
print("=" * 80)

converted_primary_mass = np.asarray(
    converted["mass"],
    dtype=float,
)

converted_system_mass = np.asarray(
    converted["systemMass"],
    dtype=float,
)

catalog_summaries = [
    {
        "catalog": "pointer-derived primary masses",
        "n": len(df_pointer_systems),
        "mass_min": df_pointer_systems["primary_mass"].min(),
        "mass_median": df_pointer_systems["primary_mass"].median(),
        "mass_max": df_pointer_systems["primary_mass"].max(),
        "mass_sum": df_pointer_systems["primary_mass"].sum(),
    },
    {
        "catalog": "pointer-derived system masses",
        "n": len(df_pointer_systems),
        "mass_min": df_pointer_systems["system_mass"].min(),
        "mass_median": df_pointer_systems["system_mass"].median(),
        "mass_max": df_pointer_systems["system_mass"].max(),
        "mass_sum": df_pointer_systems["system_mass"].sum(),
    },
    {
        "catalog": "converted primary masses",
        "n": len(converted_primary_mass),
        "mass_min": np.nanmin(converted_primary_mass),
        "mass_median": np.nanmedian(converted_primary_mass),
        "mass_max": np.nanmax(converted_primary_mass),
        "mass_sum": np.nansum(converted_primary_mass),
    },
    {
        "catalog": "converted system masses",
        "n": len(converted_system_mass),
        "mass_min": np.nanmin(converted_system_mass),
        "mass_median": np.nanmedian(converted_system_mass),
        "mass_max": np.nanmax(converted_system_mass),
        "mass_sum": np.nansum(converted_system_mass),
    },
]

for label, df in [
    ("all unresolved", df_all_unresolved),
    ("bound unresolved", df_bound_unresolved),
]:
    if "mass" in df.columns:
        masses = np.asarray(df["mass"], dtype=float)

        catalog_summaries.append(
            {
                "catalog": label,
                "n": len(masses),
                "mass_min": np.nanmin(masses),
                "mass_median": np.nanmedian(masses),
                "mass_max": np.nanmax(masses),
                "mass_sum": np.nansum(masses),
            }
        )

df_catalog_summaries = pd.DataFrame(
    catalog_summaries
)

display(df_catalog_summaries)


print()
print("=" * 80)
print("NEAREST-MASS COVERAGE AGAINST CONVERTED TABLE")
print("=" * 80)


def nearest_mass_diagnostics(
    source_masses,
    target_masses,
    source_label,
    target_label,
):
    source = np.asarray(source_masses, dtype=float)
    target = np.asarray(target_masses, dtype=float)

    differences = np.array(
        [
            np.min(np.abs(target - mass))
            for mass in source
        ],
        dtype=float,
    )

    return {
        "source": source_label,
        "target": target_label,
        "n_source": len(source),
        "fraction_within_1e-6": np.mean(
            differences <= 1.0e-6
        ),
        "fraction_within_1e-4": np.mean(
            differences <= 1.0e-4
        ),
        "median_nearest_difference": np.median(
            differences
        ),
        "maximum_nearest_difference": np.max(
            differences
        ),
    }


mass_match_results = [
    nearest_mass_diagnostics(
        converted_primary_mass,
        df_pointer_systems["primary_mass"],
        "converted primary mass",
        "pointer primary mass",
    ),
    nearest_mass_diagnostics(
        converted_system_mass,
        df_pointer_systems["system_mass"],
        "converted system mass",
        "pointer system mass",
    ),
]

if "mass" in df_all_unresolved.columns:
    mass_match_results.extend(
        [
            nearest_mass_diagnostics(
                converted_primary_mass,
                df_all_unresolved["mass"],
                "converted primary mass",
                "all unresolved mass",
            ),
            nearest_mass_diagnostics(
                converted_system_mass,
                df_all_unresolved["mass"],
                "converted system mass",
                "all unresolved mass",
            ),
        ]
    )

if "mass" in df_bound_unresolved.columns:
    mass_match_results.extend(
        [
            nearest_mass_diagnostics(
                converted_primary_mass,
                df_bound_unresolved["mass"],
                "converted primary mass",
                "bound unresolved mass",
            ),
            nearest_mass_diagnostics(
                converted_system_mass,
                df_bound_unresolved["mass"],
                "converted system mass",
                "bound unresolved mass",
            ),
        ]
    )

display(pd.DataFrame(mass_match_results))