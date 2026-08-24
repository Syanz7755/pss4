import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


COAT_RE = re.compile(r"results_coat_([0-9.+\-eE]+)$")


def apply_figure_theme() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Arial:bold",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })


def coating_from_dir(path: Path) -> float:
    match = COAT_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse coating thickness from {path.name}")
    return float(match.group(1))


def find_spectrum_csv(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("tot_P*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No tot_P*.csv file found in {result_dir}")
    return candidates[0]


def load_series(data_root: Path) -> list[dict]:
    series = []
    for result_dir in sorted(data_root.glob("results_coat_*"), key=coating_from_dir):
        csv_path = find_spectrum_csv(result_dir)
        df = pd.read_csv(csv_path)
        required = {"omega (rad/s)", "transmission"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        omega = pd.to_numeric(df["omega (rad/s)"], errors="coerce").to_numpy(float)
        transmission = pd.to_numeric(df["transmission"], errors="coerce").to_numpy(float)
        mask = np.isfinite(omega) & np.isfinite(transmission) & (transmission > 0)
        if not np.any(mask):
            raise ValueError(f"No positive finite transmission values in {csv_path}")
        series.append({
            "result_dir": result_dir,
            "csv_path": csv_path,
            "coating_um": coating_from_dir(result_dir),
            "omega": omega[mask],
            "transmission": transmission[mask],
        })
    if not series:
        raise FileNotFoundError(f"No results_coat_* directories found in {data_root}")
    return series


def plot_spectrum(series: list[dict], output_dir: Path, dpi: int) -> dict:
    apply_figure_theme()
    output_dir.mkdir(parents=True, exist_ok=True)

    coatings = np.array([item["coating_um"] for item in series], dtype=float)
    norm = Normalize(vmin=float(np.min(coatings)), vmax=float(np.max(coatings)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for item in series:
        color = cmap(norm(item["coating_um"]))
        ax.plot(
            item["omega"] / 1e14,
            item["transmission"],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
        )

    ax.set_yscale("log")
    ax.set_xlim(left=1.4, right=4.5)
    ax.set_xlabel(r"$\omega\;(10^{14}\,\mathrm{rad}\,\mathrm{s}^{-1})$", fontsize=18)
    ax.set_ylabel(r"Transmission $(\mathrm{m}^{-1})$", fontsize=18)
    ax.tick_params(axis="both", which="both", labelsize=16)
    ax.grid(True, which="major", color="0.88", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Coating thickness $t_{\mathrm{c}}$ (nm)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 6))
    cbar.set_ticklabels([f"{value * 1000:.0f}" for value in np.linspace(norm.vmin, norm.vmax, 6)])

    fig.tight_layout()
    png_path = output_dir / "combo00_H30_W1_delta600_transmission_spectrum_by_coating_viridis.png"
    pdf_path = output_dir / "combo00_H30_W1_delta600_transmission_spectrum_by_coating_viridis.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    manifest_path = output_dir / "spectrum_manifest.csv"
    rows = []
    for item in series:
        rows.append({
            "result_dir": str(item["result_dir"]),
            "csv_path": str(item["csv_path"]),
            "coating_um": item["coating_um"],
            "coating_nm": item["coating_um"] * 1000,
            "points": len(item["omega"]),
            "omega_min_rad_s": float(np.min(item["omega"])),
            "omega_max_rad_s": float(np.max(item["omega"])),
            "transmission_min": float(np.min(item["transmission"])),
            "transmission_max": float(np.max(item["transmission"])),
        })
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    return {"png": png_path, "pdf": pdf_path, "manifest": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot coating-colored transmission spectrum curves.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    series = load_series(args.data_root)
    outputs = plot_spectrum(series, args.output_dir, args.dpi)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
