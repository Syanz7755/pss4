"""Post-process saved tau(omega, k_parallel) grids by propagation regime.

The script consumes the P{id}_tau_s.csv, P{id}_tau_p.csv, and
k_parallel_grid.csv files emitted by a PSS4 run.  It does not rerun the
electromagnetic solver.
"""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fed_engine import planck_derivative
from utils import c

HBAR_EV_S = 6.582119569e-16

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


TAU_FILES = {"s": "P{probe}_tau_s.csv", "p": "P{probe}_tau_p.csv"}
OMEGA_COLUMN = "omega (rad/s)"


def read_tau_grid(path: Path):
    """Return omega, tau, and the ordered k-column names from one tau CSV."""
    frame = pd.read_csv(path)
    if OMEGA_COLUMN not in frame:
        raise ValueError(f"{path} is missing column {OMEGA_COLUMN!r}")
    k_columns = [column for column in frame.columns if column.startswith("k")]
    if not k_columns:
        raise ValueError(f"{path} has no k-index columns")
    return (frame[OMEGA_COLUMN].to_numpy(float), frame[k_columns].to_numpy(float),
            k_columns, frame)


def load_epsilon(epsilon_file: Path, omega: np.ndarray) -> np.ndarray:
    """Interpolate a three-column dielectric table on the tau frequency grid."""
    table = np.loadtxt(epsilon_file, ndmin=2)
    if table.shape[1] < 3:
        raise ValueError(f"{epsilon_file} must have omega, epsilon_real, epsilon_imag columns")
    source_omega, eps_real, eps_imag = table[:, 0], table[:, 1], table[:, 2]
    if np.any(np.diff(source_omega) <= 0):
        raise ValueError(f"{epsilon_file} must have strictly increasing omega")
    if omega[0] < source_omega[0] or omega[-1] > source_omega[-1]:
        raise ValueError("Tau frequency range lies outside the supplied dielectric table")
    return np.interp(omega, source_omega, eps_real) + 1j * np.interp(
        omega, source_omega, eps_imag)


def calculate_regime_observables(omega, tau_s, tau_p, k_parallel, k_weights,
                                 epsilon, temperature):
    """Integrate tau by polarization and prop/evan masks at every frequency."""
    boundary = np.real(np.sqrt(epsilon)) * omega / c
    prop_mask = k_parallel[None, :] < boundary[:, None]
    weighted_measure = k_parallel * k_weights / (2 * np.pi)
    result = {"omega_rad_s": omega, "k_prop_evan_1_m": boundary}

    per_pol = {"s": tau_s, "p": tau_p}
    for polarization, tau in per_pol.items():
        for regime, mask in (("prop", prop_mask), ("evan", ~prop_mask)):
            value = np.sum(tau * weighted_measure[None, :] * mask, axis=1)
            result[f"transmission_{polarization}_{regime}"] = value
        result[f"transmission_{polarization}_total"] = (
            result[f"transmission_{polarization}_prop"] +
            result[f"transmission_{polarization}_evan"])

    for regime in ("prop", "evan", "total"):
        result[f"transmission_{regime}"] = (
            result[f"transmission_s_{regime}"] + result[f"transmission_p_{regime}"])
        result[f"spectral_flux_{regime}_W_m2_K_rad_s"] = (
            np.array([planck_derivative(item, temperature) for item in omega]) *
            result[f"transmission_{regime}"] / (2 * np.pi))
    return pd.DataFrame(result)


def plot_tau(omega, k_parallel, tau, boundary, polarization, output_path):
    positive = tau[tau > 0]
    if positive.size == 0:
        raise ValueError(f"tau_{polarization} contains no positive values")
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    energy_ev = HBAR_EV_S * omega
    mesh = ax.pcolormesh(k_parallel, energy_ev, np.ma.masked_less_equal(tau, 0),
                         shading="auto", cmap="RdBu",
                         norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())))
    # A 2502 x 1000 mesh would otherwise become hundreds of MB of individual
    # vector cells.  Rasterize only the heatmap; axes, text, and boundary stay
    # vector-sharp in SVG/PDF.
    mesh.set_rasterized(True)
    fig.colorbar(mesh, ax=ax, label=rf"$\tau_{polarization}$")
    ax.plot(boundary, energy_ev, color="white", linestyle="--", linewidth=1.4,
            label="prop–evan boundary")
    ax.set(xscale="log", yscale="log", xlabel=r"$k_{\parallel}$ (m$^{-1}$)",
           ylabel=r"$\hbar\omega$ (eV)",
           title=rf"$\tau_{polarization}(\omega, k_{{\parallel}})$")
    ax.legend(loc="lower right")
    save_figure_formats(fig, output_path)
    plt.close(fig)


def save_figure_formats(fig, png_path):
    """Write publication and raster forms of one figure from a PNG base path."""
    for suffix in (".png", ".svg", ".pdf"):
        options = {"bbox_inches": "tight"}
        if suffix == ".png":
            options["dpi"] = 300
        fig.savefig(png_path.with_suffix(suffix), **options)


def plot_observables(table, output_dir):
    omega = table["omega_rad_s"]
    for quantity, ylabel, suffix in (
        ("transmission", r"Transmission density (m$^{-2}$)", "transmission"),
        ("spectral_flux", r"Spectral flux (W m$^{-2}$ K$^{-1}$ / (rad s$^{-1}$))",
         "spectral_flux"),
    ):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for regime, color in (("total", "black"), ("prop", "tab:blue"), ("evan", "tab:red")):
            column = (f"{quantity}_{regime}" if quantity == "transmission" else
                      f"{quantity}_{regime}_W_m2_K_rad_s")
            ax.plot(omega, table[column], label=regime, color=color, linewidth=1.5)
        ax.set(xscale="log", yscale="log", xlabel=r"$\omega$ (rad s$^{-1}$)", ylabel=ylabel,
               title=f"{suffix}: total and prop/evan contributions")
        ax.legend()
        ax.grid(True, which="both", alpha=0.2)
        save_figure_formats(fig, output_dir / f"{suffix}_prop_evan.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Directory containing P*_tau_{s,p}.csv")
    parser.add_argument("--probe", required=True, help="Probe id, e.g. 2 for P2_tau_p.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in K")
    parser.add_argument("--epsilon-file", type=Path,
                        default=PROJECT_DIR / "materials" / "aligned_Vacuum.txt",
                        help="Three-column gap-medium dielectric table (default: Vacuum)")
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    result_dir, output_dir = args.result_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_paths = {pol: result_dir / name.format(probe=args.probe) for pol, name in TAU_FILES.items()}
    for path in [*tau_paths.values(), result_dir / "k_parallel_grid.csv", args.epsilon_file]:
        if not path.is_file():
            parser.error(f"Required file not found: {path}")

    omega_s, tau_s, k_columns_s, _ = read_tau_grid(tau_paths["s"])
    omega_p, tau_p, k_columns_p, _ = read_tau_grid(tau_paths["p"])
    if not np.array_equal(omega_s, omega_p) or k_columns_s != k_columns_p:
        raise ValueError("tau_s and tau_p grids must have identical omega and k columns")
    k_frame = pd.read_csv(result_dir / "k_parallel_grid.csv")
    if len(k_frame) != len(k_columns_s):
        raise ValueError("k_parallel_grid.csv length does not match tau grid columns")
    k_parallel = k_frame["k_parallel (1/m)"].to_numpy(float)
    k_weights = k_frame["k_weight (1/m)"].to_numpy(float)
    epsilon = load_epsilon(args.epsilon_file, omega_s)
    summary = calculate_regime_observables(omega_s, tau_s, tau_p, k_parallel, k_weights,
                                           epsilon, args.temperature)
    summary.to_csv(output_dir / "prop_evan_observables.csv", index=False)

    raw_dir = output_dir / "raw_input_csv"
    raw_dir.mkdir(exist_ok=True)
    for source in [tau_paths["s"], tau_paths["p"], result_dir / "k_parallel_grid.csv", args.epsilon_file]:
        shutil.copy2(source, raw_dir / source.name)
    pd.DataFrame({"omega_rad_s": omega_s, "epsilon_real": epsilon.real,
                  "epsilon_imag": epsilon.imag,
                  "k_prop_evan_1_m": summary["k_prop_evan_1_m"]}).to_csv(
        output_dir / "prop_evan_boundary.csv", index=False)

    boundary = summary["k_prop_evan_1_m"].to_numpy(float)
    plot_tau(omega_s, k_parallel, tau_s, boundary, "s", output_dir / "tau_s_RdBu_prop_evan.png")
    plot_tau(omega_s, k_parallel, tau_p, boundary, "p", output_dir / "tau_p_RdBu_prop_evan.png")
    plot_observables(summary, output_dir)
    print(f"Wrote prop/evan analysis to {output_dir}")


if __name__ == "__main__":
    main()
