"""Recalculate the six 300 K heff-vs-H/W curves on a fixed aspect-ratio range.

The local sidewall hW values are reused from the corrected July PSS4 summaries,
matching the workflow in ``D:/Researches/wqs_comps/scripts/
extend_hw_until_saturation.py``. Only the downstream device-level HT model is
reevaluated as H/W changes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WQS_ROOT = PROJECT_ROOT.parent
SHARED_SCRIPTS = WQS_ROOT / "scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

import extend_hw_until_saturation as extension  # noqa: E402


DEFAULT_BASE_CSVS = [
    PROJECT_ROOT
    / "combined_delta0p600_HW3to100_heff_plots_20260707_corrected_semilogy"
    / "combined_row_level_heff_inputs_corrected.csv",
    PROJECT_ROOT
    / "combined_delta0p600_HW3to100_heff_temperature_plots_20260707_corrected_semilogy"
    / "combined_row_level_heff_inputs_by_temperature_corrected.csv",
]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "combined_delta0p600_HW1to5000_T300K_d1um_htonly_loglog_wstyle_20260729"
)

COLORS = {
    "no_coat": "#1f77b4",
    "coat_200nm": "#d62728",
    "delta100_no_coat": "#2ca02c",
    "delta200_coat50_gap100": "#9467bd",
}
LINESTYLES = {0.5: "-", 0.6667: "--", 1.0: "-."}
WIDTH_LABELS = {0.5: "W=0.5 μm", 0.6667: "W=0.667 μm", 1.0: "W=1.0 μm"}
CASE_LABELS = {
    "no_coat": "600/0/600",
    "coat_200nm": "600/200/200",
    "delta100_no_coat": "100/0/100",
    "delta200_coat50_gap100": "200/50/100",
}
DELTA100_RESULT_ROOT = (
    PROJECT_ROOT / "delta100_no_coat_hW_300K_20260729" / "results"
)
DELTA100_TOT_FILES = {
    0.5: DELTA100_RESULT_ROOT / "W0p5_gap100_coat000" / "tot_P1.csv",
    0.6667: DELTA100_RESULT_ROOT / "W0p6667_gap100_coat000" / "tot_P1.csv",
    1.0: DELTA100_RESULT_ROOT / "W1_gap100_coat000" / "tot_P1.csv",
}
DELTA200_COAT50_RESULT_ROOT = (
    PROJECT_ROOT / "delta200_coat50_gap100_hW_300K_20260729" / "results"
)
DELTA200_COAT50_TOT_FILES = {
    0.5: DELTA200_COAT50_RESULT_ROOT / "W0p5_gap100_coat050" / "tot_P2.csv",
    0.6667: DELTA200_COAT50_RESULT_ROOT
    / "W0p6667_gap100_coat050"
    / "tot_P2.csv",
    1.0: DELTA200_COAT50_RESULT_ROOT / "W1_gap100_coat050" / "tot_P2.csv",
}

ROW_COLUMNS = [
    "temperature_K",
    "cfg_id",
    "W_um",
    "aspect_ratio",
    "H_um",
    "delta_um",
    "d_um",
    "d_eff_um",
    "coat_um",
    "coat_label",
    "hW_W_m2_K",
    "heff_direct_W_m2_K",
    "heff_HT_W_m2_K",
    "output_normalization_factor",
    "tot_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-aspect", type=float, default=1.0)
    parser.add_argument("--max-aspect", type=float, default=5000.0)
    parser.add_argument("--dense-through", type=float, default=100.0)
    parser.add_argument("--dense-step", type=float, default=1.0)
    parser.add_argument("--sparse-step", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--base-row-csv", type=Path, action="append", default=None)
    return parser.parse_args()


def value_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive")
    count = max(int(math.floor((stop - start) / step + 1e-12)), 0)
    values = [start + i * step for i in range(count + 1)]
    if not values or values[-1] < stop - 1e-12:
        values.append(stop)
    return values


def aspect_grid(args: argparse.Namespace) -> list[float]:
    low_stop = min(float(args.max_aspect), float(args.dense_through))
    values = value_grid(float(args.min_aspect), low_stop, float(args.dense_step))
    if args.max_aspect > args.dense_through:
        sparse_start = (
            math.floor(float(args.dense_through) / float(args.sparse_step)) + 1
        ) * float(args.sparse_step)
        values.extend(
            value_grid(sparse_start, float(args.max_aspect), float(args.sparse_step))
        )
    return sorted(set(round(value, 12) for value in values))


def aspect_slug(aspect: float) -> str:
    if float(aspect).is_integer():
        return f"{int(round(aspect)):05d}"
    return extension.slug_number(aspect)


def integrated_h_w(path: Path) -> float:
    spectrum = pd.read_csv(path)
    return float(
        np.trapezoid(
            spectrum["spectral_flux (W/m2/K/(rad/s))"].to_numpy(dtype=float),
            spectrum["omega (rad/s)"].to_numpy(dtype=float),
        )
    )


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    temperature = int(round(float(args.temperature)))
    base_paths = [path.resolve() for path in (args.base_row_csv or DEFAULT_BASE_CSVS)]
    base_rows = extension.load_base_rows(base_paths)
    sources = extension.source_lookup(base_rows, [temperature])
    delta100_sources = {
        width: {
            "hW": integrated_h_w(path),
            "tot_file": path.relative_to(PROJECT_ROOT).as_posix(),
            "output_normalization_factor": 1.0,
        }
        for width, path in DELTA100_TOT_FILES.items()
    }
    delta200_coat50_sources = {
        width: {
            "hW": integrated_h_w(path),
            "tot_file": path.relative_to(PROJECT_ROOT).as_posix(),
            "output_normalization_factor": 1.0,
        }
        for width, path in DELTA200_COAT50_TOT_FILES.items()
    }
    cases = [
        {**case, "delta_um": extension.DELTA_UM}
        for case in extension.CASES
    ] + [
        {
            "case_key": "gap100_coat000",
            "coat_label": "delta100_no_coat",
            "coating_thickness_um": 0.0,
            "d_eff_um": 0.1,
            "delta_um": 0.1,
        },
        {
            "case_key": "gap100_coat050",
            "coat_label": "delta200_coat50_gap100",
            "coating_thickness_um": 0.05,
            "d_eff_um": 0.1,
            "delta_um": 0.2,
        },
    ]
    rows: list[dict[str, object]] = []

    for aspect in aspect_grid(args):
        for width in extension.WIDTHS_UM:
            height_um = width * aspect
            for case in cases:
                if case["coat_label"] == "delta100_no_coat":
                    source = delta100_sources[width]
                elif case["coat_label"] == "delta200_coat50_gap100":
                    source = delta200_coat50_sources[width]
                else:
                    source = sources[(temperature, width, case["coat_label"])]
                h_w = float(source["hW"])
                direct = extension.direct_heff(
                    h_w,
                    height_um,
                    extension.ASSUMED_D_UM,
                    width,
                    case["delta_um"],
                )
                ht = extension.monotonic_ht_metrics(
                    H_um=height_um,
                    d_um=extension.ASSUMED_D_UM,
                    W_um=width,
                    delta_um=case["delta_um"],
                    coat_um=case["coating_thickness_um"],
                    hW=h_w,
                    temperature_K=temperature,
                )
                cfg_id = (
                    f"W{extension.slug_number(width)}_AR{aspect_slug(aspect)}_"
                    f"H{extension.slug_number(height_um)}_{case['case_key']}"
                )
                rows.append(
                    {
                        "temperature_K": temperature,
                        "cfg_id": cfg_id,
                        "W_um": width,
                        "aspect_ratio": aspect,
                        "H_um": height_um,
                        "delta_um": case["delta_um"],
                        "d_um": extension.ASSUMED_D_UM,
                        "d_eff_um": case["d_eff_um"],
                        "coat_um": case["coating_thickness_um"],
                        "coat_label": case["coat_label"],
                        "hW_W_m2_K": h_w,
                        "heff_direct_W_m2_K": direct,
                        "heff_HT_W_m2_K": ht["heff"],
                        "output_normalization_factor": source[
                            "output_normalization_factor"
                        ],
                        "tot_file": source["tot_file"],
                    }
                )

    return (
        pd.DataFrame(rows, columns=ROW_COLUMNS)
        .sort_values(
            [
                "temperature_K",
                "coat_label",
                "W_um",
                "aspect_ratio",
                "output_normalization_factor",
            ]
        )
        .reset_index(drop=True)
    )


def build_curve_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        common = {
            "temperature_K": int(row.temperature_K),
            "coating": row.coat_label,
            "W_um": float(row.W_um),
            "H_over_W": float(row.aspect_ratio),
            "H_um": float(row.H_um),
            "delta_um": float(row.delta_um),
            "d_um": float(row.d_um),
            "d_eff_um": float(row.d_eff_um),
            "coat_um": float(row.coat_um),
            "hW_W_m2_K": float(row.hW_W_m2_K),
            "output_normalization_factor": float(row.output_normalization_factor),
            "source_tot_file": row.tot_file,
        }
        rows.append(
            {
                **common,
                "method": "HT_model",
                "heff_W_m2_K": float(row.heff_HT_W_m2_K),
            }
        )
        rows.append(
            {
                **common,
                "method": "direct",
                "heff_W_m2_K": float(row.heff_direct_W_m2_K),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["temperature_K", "method", "coating", "W_um", "H_over_W"]
    )


def apply_figure_theme() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def plot_one(
    curves: pd.DataFrame,
    output_dir: Path,
    temperature: int,
    method: str,
    min_aspect: float,
    max_aspect: float,
) -> Path:
    subset = curves.loc[
        (curves["temperature_K"] == temperature)
        & (curves["method"] == method)
        & np.isfinite(curves["heff_W_m2_K"])
        & (curves["heff_W_m2_K"] > 0)
    ]
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=180)
    for coating in (
        "no_coat",
        "coat_200nm",
        "delta100_no_coat",
        "delta200_coat50_gap100",
    ):
        for width in (0.5, 0.6667, 1.0):
            group = subset.loc[
                (subset["coating"] == coating)
                & np.isclose(subset["W_um"], width, atol=5e-4)
            ].sort_values("H_over_W")
            ax.loglog(
                group["H_over_W"],
                group["heff_W_m2_K"],
                color=COLORS[coating],
                linestyle=LINESTYLES[width],
                linewidth=2.0,
                label=f"{CASE_LABELS[coating]}, {WIDTH_LABELS[width]}",
            )

    ax.set_xlim(min_aspect, max_aspect)
    ax.set_xlabel(r"$H/W$")
    ax.set_ylabel(r"$h_{\mathrm{eff}}\;(\mathrm{W\,m^{-2}\,K^{-1}})$")
    title_method = "HT model" if method == "HT_model" else "Direct"
    ax.set_title(f"{title_method}: effective heat transfer vs H/W at {temperature} K")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(
        fontsize=9.5,
        ncol=4,
        frameon=True,
        title="before/coat/gap (nm); coating thickness is per side",
        title_fontsize=10.5,
    )
    fig.tight_layout()
    safe_method = "HT_model" if method == "HT_model" else "direct"
    output_path = (
        output_dir / f"heff_{safe_method}_vs_HW_{temperature}K_loglog_wstyle.png"
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_figure_theme()

    data = build_rows(args)
    curves = build_curve_rows(data)
    row_csv = output_dir / "combined_row_level_heff_inputs_loglog_wstyle.csv"
    curve_csv = output_dir / "heff_curves_vs_aspect_ratio_loglog_wstyle.csv"
    data.to_csv(row_csv, index=False)
    curves.to_csv(curve_csv, index=False)

    temperature = int(round(float(args.temperature)))
    images = [
        plot_one(
            curves,
            output_dir,
            temperature,
            method,
            float(args.min_aspect),
            float(args.max_aspect),
        )
        for method in ("HT_model", "direct")
    ]
    print(f"aspect_points={data['aspect_ratio'].nunique()} row_count={len(data)}")
    print(f"wrote {row_csv}")
    print(f"wrote {curve_csv}")
    for image in images:
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
