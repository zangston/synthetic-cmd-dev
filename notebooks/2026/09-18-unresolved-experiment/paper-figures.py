from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT = Path.cwd()

CACHE_DIR = ROOT / "binary_resolution_seed00_cmd_cache"
OUTPUT_DIR = ROOT / "binary_resolution_seed00_cmd_outputs"
PAPER_DIR = ROOT / "paper_figure_outputs"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected files exist:\n"
        + "\n".join(f"  {p}" for p in paths)
    )


METRICS_PATH = first_existing(
    CACHE_DIR / "spread_summary.csv",
    OUTPUT_DIR / "spread_metrics_binary_resolution.csv",
)

metrics = pd.read_csv(METRICS_PATH)


# Prefer the already-generated fraction table.
fraction_path = OUTPUT_DIR / "binary_resolution_fractions.csv"

if fraction_path.exists():
    fractions = pd.read_csv(fraction_path)

else:
    # Fallback: reconstruct fractions from the cached resolution-state history.
    history_path = first_existing(
        OUTPUT_DIR / "binary_resolution_state_history.csv",
    )
    history = pd.read_csv(history_path)

    fractions = (
        history
        .groupby(
            ["snapshot_time_myr", "diagram", "projection"],
            as_index=False,
        )
        .agg(
            fraction_unresolved=("unresolved", "mean"),
            n_binaries=("pair_id", "nunique"),
        )
    )


# ----------------------------------------------------------------------
# Labels / ordering
# ----------------------------------------------------------------------

DIAGRAMS = [
    (
        "f070w_f200w",
        "JWST F070W - F200W vs. F200W",
    ),
    (
        "f182m_f200w",
        "JWST F182M - F200W vs. F200W",
    ),
    (
        "hst_f555w_f814w",
        "HST F555W - F814W vs. F814W",
    ),
]

MODE_STYLE = {
    "all_resolved": {
        "label": "All resolved",
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.2,
        "marker": None,
        "zorder": 3,
    },
    "all_unresolved": {
        "label": "All unresolved",
        "color": "firebrick",
        "linestyle": "-",
        "linewidth": 2.2,
        "marker": None,
        "zorder": 3,
    },
    "los_xy": {
        "label": "Rayleigh: XY",
        "color": "tab:blue",
        "linestyle": "--",
        "linewidth": 1.5,
        "marker": "o",
        "zorder": 4,
    },
    "los_xz": {
        "label": "Rayleigh: XZ",
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 1.5,
        "marker": "s",
        "zorder": 4,
    },
    "los_yz": {
        "label": "Rayleigh: YZ",
        "color": "tab:green",
        "linestyle": "--",
        "linewidth": 1.5,
        "marker": "^",
        "zorder": 4,
    },
}

PROJECTION_STYLE = {
    "XY": ("tab:blue", "o"),
    "XZ": ("tab:orange", "s"),
    "YZ": ("tab:green", "^"),
}


# ----------------------------------------------------------------------
# Main 2 x 3 paper figure
# ----------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15.0, 8.0),
    sharex="col",
    constrained_layout=True,
)

for col, (diagram_key, title) in enumerate(DIAGRAMS):

    # --------------------------------------------------------------
    # Top row: spread metric
    # --------------------------------------------------------------
    ax = axes[0, col]

    d = metrics[metrics["diagram"] == diagram_key].copy()

    for mode, style in MODE_STYLE.items():
        sub = (
            d[d["resolution_mode"] == mode]
            .sort_values("snapshot_time_myr")
        )

        if sub.empty:
            continue

        ax.plot(
            sub["snapshot_time_myr"],
            sub["spread_metric"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=3.2,
            markevery=2,
            zorder=style["zorder"],
        )

    ax.set_title(title)
    ax.grid(alpha=0.22)

    if col == 0:
        ax.set_ylabel(
            "Mean quartile-tail spread [mag]"
        )

    # --------------------------------------------------------------
    # Bottom row: fraction unresolved
    # --------------------------------------------------------------
    ax = axes[1, col]

    f = fractions[
        fractions["diagram"] == diagram_key
    ].copy()

    for projection, (color, marker) in PROJECTION_STYLE.items():
        sub = (
            f[f["projection"].astype(str).str.upper() == projection]
            .sort_values("snapshot_time_myr")
        )

        if sub.empty:
            continue

        ax.plot(
            sub["snapshot_time_myr"],
            sub["fraction_unresolved"],
            color=color,
            marker=marker,
            markersize=3.5,
            markevery=2,
            linewidth=1.6,
            label=projection,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cluster time [Myr]")
    ax.grid(alpha=0.22)

    if col == 0:
        ax.set_ylabel(
            "Fraction of current binaries unresolved"
        )


# Shared legend for spread modes.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.055),
)

fig.suptitle(
    "Effect of Binary Angular Resolution on Synthetic CMD Spread",
    y=1.07,
)

png_path = PAPER_DIR / "fig_binary_resolution_summary.png"
pdf_path = PAPER_DIR / "fig_binary_resolution_summary.pdf"

fig.savefig(
    png_path,
    dpi=320,
    bbox_inches="tight",
)
fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

print("Saved:", png_path)
print("Saved:", pdf_path)

plt.show()


# ----------------------------------------------------------------------
# Quantify the results for manuscript text
# ----------------------------------------------------------------------

# Put the five spread modes into columns at each diagram/time.
spread_wide = (
    metrics
    .pivot_table(
        index=["diagram", "snapshot_time_myr"],
        columns="resolution_mode",
        values="spread_metric",
        aggfunc="first",
    )
    .reset_index()
)

records = []

for diagram_key, title in DIAGRAMS:

    d = spread_wide[
        spread_wide["diagram"] == diagram_key
    ].copy()

    required = {
        "all_resolved",
        "all_unresolved",
        "los_xy",
        "los_xz",
        "los_yz",
    }

    if not required.issubset(d.columns):
        print(
            f"Skipping manuscript statistics for {diagram_key}; "
            "not all five modes are present."
        )
        continue

    extreme_offset = (
        d["all_unresolved"] - d["all_resolved"]
    )

    los_array = d[
        ["los_xy", "los_xz", "los_yz"]
    ].to_numpy(float)

    los_projection_range = (
        np.nanmax(los_array, axis=1)
        - np.nanmin(los_array, axis=1)
    )

    for projection, mode in [
        ("XY", "los_xy"),
        ("XZ", "los_xz"),
        ("YZ", "los_yz"),
    ]:

        frac = fractions[
            (fractions["diagram"] == diagram_key)
            & (
                fractions["projection"]
                .astype(str)
                .str.upper()
                == projection
            )
        ][
            [
                "snapshot_time_myr",
                "fraction_unresolved",
            ]
        ].copy()

        tmp = d[
            [
                "snapshot_time_myr",
                "all_resolved",
                "all_unresolved",
                mode,
            ]
        ].merge(
            frac,
            on="snapshot_time_myr",
            how="inner",
        )

        denom = (
            tmp["all_unresolved"]
            - tmp["all_resolved"]
        )

        good = (
            np.isfinite(denom)
            & (np.abs(denom) > 1e-12)
        )

        relative_position = np.full(
            len(tmp),
            np.nan,
            dtype=float,
        )

        relative_position[good] = (
            (
                tmp.loc[good, mode]
                - tmp.loc[good, "all_resolved"]
            )
            / denom[good]
        )

        los_offset = (
            tmp[mode]
            - tmp["all_resolved"]
        )

        records.append(
            {
                "diagram": diagram_key,
                "diagram_title": title,
                "projection": projection,

                "mean_fraction_unresolved":
                    tmp["fraction_unresolved"].mean(),

                "min_fraction_unresolved":
                    tmp["fraction_unresolved"].min(),

                "max_fraction_unresolved":
                    tmp["fraction_unresolved"].max(),

                "mean_all_unresolved_minus_resolved_spread":
                    extreme_offset.mean(),

                "std_all_unresolved_minus_resolved_spread":
                    extreme_offset.std(ddof=1),

                "mean_los_minus_resolved_spread":
                    los_offset.mean(),

                "std_los_minus_resolved_spread":
                    los_offset.std(ddof=1),

                # 0 = resolved limit
                # 1 = unresolved limit
                "mean_los_position_between_extremes":
                    np.nanmean(relative_position),

                "std_los_position_between_extremes":
                    np.nanstd(relative_position, ddof=1),

                "mean_spread_range_across_viewing_axes":
                    np.nanmean(los_projection_range),

                "max_spread_range_across_viewing_axes":
                    np.nanmax(los_projection_range),
            }
        )


summary = pd.DataFrame(records)

summary_path = (
    PAPER_DIR
    / "binary_resolution_paper_summary_statistics.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 90)
print("BINARY-RESOLUTION MANUSCRIPT SUMMARY")
print("=" * 90)
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:", summary_path)