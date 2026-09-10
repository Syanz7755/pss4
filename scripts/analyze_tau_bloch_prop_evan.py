"""Bloch-mode prop/evan post-processing for a saved PSS4 tau grid.

Unlike the gap light-line post-processor, this script classifies the actual
periodic unit cell declared by ``input_stack``.  The two classifications answer
different physical questions and are deliberately written to separate folders.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analyze_tau_prop_evan import (HBAR_EV_S, OMEGA_COLUMN, read_tau_grid, save_figure_formats)
from config_parser import parse_stack_file
from fed_engine import planck_derivative
from materials import load_materials_and_grid
from tau_grid import compute_unit_cell_s_matrix
from utils import c


def forward_bloch_k(s_matrix, period_m):
    """Return the forward/decaying Bloch wavevector from a scalar S matrix.

    The transfer matrix eigenvalues satisfy lambda=exp(i*K_B*period).  The
    eigenvalue inside the unit circle is selected, giving Im(K_B)>=0.
    """
    r11, t12, t21, r22 = s_matrix
    if abs(t12) < 1e-14:
        return np.nan + 1j * np.nan
    transfer = np.array(((-r11 / t12, 1.0 / t12),
                         (t21 - r22 * r11 / t12, r22 / t12)), dtype=complex)
    eigenvalues = np.linalg.eigvals(transfer)
    selected = eigenvalues[np.argmin(np.abs(eigenvalues))]
    return -1j * np.log(selected) / period_m


def classify_bloch(layers, omega, k_parallel, material_eps, eta):
    """Compute K_B and prop masks for every saved (omega,k,polarization)."""
    period_m = sum(layer["thickness"] for layer in layers)
    if period_m <= 0:
        raise ValueError("The input stack has a non-positive unit-cell period")
    shape = (len(omega), len(k_parallel))
    k_bloch = {pol: np.empty(shape, dtype=complex) for pol in ("s", "p")}
    masks = {pol: np.zeros(shape, dtype=bool) for pol in ("s", "p")}
    for freq_idx, frequency in enumerate(omega):
        for polarization in ("s", "p"):
            for k_idx, k_value in enumerate(k_parallel):
                k_value_bloch = forward_bloch_k(
                    compute_unit_cell_s_matrix(layers, frequency, k_value, polarization,
                                               material_eps, freq_idx), period_m)
                k_bloch[polarization][freq_idx, k_idx] = k_value_bloch
                attenuation_limit = eta * max(abs(k_value_bloch.real), frequency / c)
                masks[polarization][freq_idx, k_idx] = (
                    np.isfinite(k_value_bloch) and abs(k_value_bloch.imag) <= attenuation_limit)
    return k_bloch, masks, period_m


def integrate_by_mask(omega, tau_by_pol, k_parallel, k_weights, masks, temperature):
    measure = k_parallel * k_weights / (2 * np.pi)
    result = {"omega_rad_s": omega}
    for polarization, tau in tau_by_pol.items():
        for regime, mask in (("prop", masks[polarization]), ("evan", ~masks[polarization])):
            result[f"transmission_{polarization}_{regime}"] = np.sum(
                tau * measure[None, :] * mask, axis=1)
        result[f"transmission_{polarization}_total"] = (
            result[f"transmission_{polarization}_prop"] + result[f"transmission_{polarization}_evan"])
    dtheta = np.array([planck_derivative(value, temperature) for value in omega])
    for regime in ("prop", "evan", "total"):
        result[f"transmission_{regime}"] = (
            result[f"transmission_s_{regime}"] + result[f"transmission_p_{regime}"])
        result[f"spectral_flux_{regime}_W_m2_K_rad_s"] = (
            dtheta * result[f"transmission_{regime}"] / (2 * np.pi))
    return pd.DataFrame(result)


def plot_bloch_tau(omega, k_parallel, tau, mask, polarization, output_path):
    from matplotlib.colors import LogNorm
    positive = tau[tau > 0]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    energy_ev = HBAR_EV_S * omega
    mesh = ax.pcolormesh(k_parallel, energy_ev, np.ma.masked_less_equal(tau, 0),
                         shading="auto", cmap="RdBu",
                         norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())))
    mesh.set_rasterized(True)
    fig.colorbar(mesh, ax=ax, label=rf"$\tau_{polarization}$")
    # Transparent grey hatch-free overlay: dark = Bloch-evanescent, clear = propagating.
    ax.pcolormesh(k_parallel, energy_ev, (~mask).astype(float), shading="auto",
                  cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 0.24)]), vmin=0, vmax=1,
                  rasterized=True)
    ax.contour(k_parallel, energy_ev, mask.astype(float), levels=[0.5], colors="white",
               linestyles="--", linewidths=1.2)
    ax.set(xscale="log", yscale="log", xlabel=r"$k_{\parallel}$ (m$^{-1}$)",
           ylabel=r"$\hbar\omega$ (eV)",
           title=rf"$\tau_{polarization}(\omega,k_{{\parallel}})$: Bloch prop/evan")
    save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_observables(table, output_dir):
    for quantity, ylabel in (("transmission", r"Transmission density (m$^{-2}$)"),
                              ("spectral_flux", r"Spectral flux (W m$^{-2}$ K$^{-1}$ / (rad s$^{-1}$))")):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for regime, color in (("total", "black"), ("prop", "tab:blue"), ("evan", "tab:red")):
            column = f"{quantity}_{regime}" if quantity == "transmission" else f"{quantity}_{regime}_W_m2_K_rad_s"
            ax.plot(table["omega_rad_s"], table[column], label=regime, color=color, linewidth=1.5)
        ax.set(xscale="log", yscale="log", xlabel=r"$\omega$ (rad s$^{-1}$)", ylabel=ylabel,
               title=f"Bloch {quantity}: total and prop/evan contributions")
        ax.legend(); ax.grid(True, which="both", alpha=0.2)
        save_figure_formats(fig, output_dir / f"bloch_{quantity}_prop_evan.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("input_stack", type=Path, help="Exact PSS4 periodic-stack input used for the tau run")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--eta", type=float, default=0.01,
                        help="Bloch propagation threshold; default 0.01")
    args = parser.parse_args()
    if args.temperature <= 0 or args.eta <= 0:
        parser.error("--temperature and --eta must be positive")
    result_dir, output_dir = args.result_dir.resolve(), args.output_dir.resolve()
    tau_s_path, tau_p_path = (result_dir / f"P{args.probe}_tau_s.csv",
                              result_dir / f"P{args.probe}_tau_p.csv")
    omega_s, tau_s, columns_s, _ = read_tau_grid(tau_s_path)
    omega_p, tau_p, columns_p, _ = read_tau_grid(tau_p_path)
    if not np.array_equal(omega_s, omega_p) or columns_s != columns_p:
        raise ValueError("tau_s and tau_p grids must be identical")
    k_grid = pd.read_csv(result_dir / "k_parallel_grid.csv")
    k_parallel = k_grid["k_parallel (1/m)"].to_numpy(float)
    k_weights = k_grid["k_weight (1/m)"].to_numpy(float)
    layers, _ = parse_stack_file(str(args.input_stack))
    model_omega, material_eps = load_materials_and_grid(layers, all_freq=True)
    if not np.array_equal(omega_s, model_omega):
        raise ValueError("Saved tau frequencies do not exactly match the input stack material grid")
    k_bloch, masks, period_m = classify_bloch(layers, omega_s, k_parallel, material_eps, args.eta)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = integrate_by_mask(omega_s, {"s": tau_s, "p": tau_p}, k_parallel, k_weights,
                                masks, args.temperature)
    summary.to_csv(output_dir / "bloch_prop_evan_observables.csv", index=False)
    pd.DataFrame({OMEGA_COLUMN: omega_s, "period_m": period_m, "eta": args.eta,
                  "bloch_propagating_fraction_s": masks["s"].mean(axis=1),
                  "bloch_propagating_fraction_p": masks["p"].mean(axis=1)}).to_csv(
        output_dir / "bloch_prop_evan_summary.csv", index=False)
    for pol, tau in (("s", tau_s), ("p", tau_p)):
        plot_bloch_tau(omega_s, k_parallel, tau, masks[pol], pol,
                       output_dir / f"tau_{pol}_RdBu_bloch_prop_evan.png")
    plot_observables(summary, output_dir)
    print(f"Wrote Bloch prop/evan analysis to {output_dir}; period={period_m:.6e} m, eta={args.eta}")


if __name__ == "__main__":
    main()
