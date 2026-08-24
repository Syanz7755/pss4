#!/usr/bin/env python3
"""Run and summarize a continuous symmetric SiO2 coating sweep with pss4."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
OMEGA_COLUMN = "omega (rad/s)"
SPECTRAL_COLUMN = "spectral_flux (W/m2/K/(rad/s))"


def coating_values(start_nm: int, stop_nm: int, step_nm: int) -> list[int]:
    if step_nm <= 0 or start_nm < 0 or stop_nm < start_nm:
        raise ValueError("Require 0 <= coat-start-nm <= coat-stop-nm and coat-step-nm > 0")
    values = list(range(start_nm, stop_nm + 1, step_nm))
    if values[-1] != stop_nm:
        raise ValueError("coat-stop-nm must lie exactly on the requested coating grid")
    return values


def stack_layers(w_um: float, delta_nm: float, coat_nm: float) -> list[tuple[str, float, bool, bool]]:
    gap_nm = delta_nm - 2.0 * coat_nm
    if w_um <= 0 or gap_nm <= 0:
        raise ValueError("W must be positive and delta - 2*t_c must remain positive")
    layers: list[tuple[str, float, bool, bool]] = [("SiN", w_um / 2.0, True, False)]
    if coat_nm > 0:
        layers.append(("SiO2", coat_nm / 1000.0, True, False))
    layers.append(("Vacuum", gap_nm / 1000.0, False, True))
    if coat_nm > 0:
        layers.append(("SiO2", coat_nm / 1000.0, True, False))
    layers.append(("SiN", w_um / 2.0, True, False))
    return layers


def stack_text(w_um: float, delta_nm: float, coat_nm: float) -> str:
    gap_nm = delta_nm - 2.0 * coat_nm
    lines = [
        f"# W={w_um:.6f} um, delta={delta_nm:.6f} nm, t_c={coat_nm:.6f} nm",
        f"# Symmetric coating convention: d_eff=delta-2*t_c={gap_nm:.6f} nm",
        "# layer_id,material,thickness_um,is_source,is_probe",
        "PERIODIC_BOUNDARY",
    ]
    for layer_id, (material, thickness_um, source, probe) in enumerate(
        stack_layers(w_um, delta_nm, coat_nm)
    ):
        lines.append(
            f"{layer_id},{material},{thickness_um:.9f},{str(source)},{str(probe)}"
        )
    lines.append("PERIODIC_BOUNDARY")
    return "\n".join(lines) + "\n"


def case_name(coat_nm: int) -> str:
    return f"coat_{coat_nm:03d}nm"


def make_run_root(output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_DIR / "analysis" /
            f"pss4_W1um_delta100nm_SiN_SiO2coat_0to45nm_1nm_300K_{stamp}").resolve()


def prepare_inputs(run_root: Path, w_um: float, delta_nm: int,
                   coatings: Sequence[int], temperature: float) -> pd.DataFrame:
    inputs_dir = run_root / "inputs"
    results_dir = run_root / "results"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for coat_nm in coatings:
        name = case_name(coat_nm)
        input_path = inputs_dir / f"{name}.txt"
        input_path.write_text(stack_text(w_um, delta_nm, coat_nm), encoding="utf-8")
        records.append({
            "case": name,
            "W_um": w_um,
            "delta_nm": delta_nm,
            "coating_nm": coat_nm,
            "effective_vacuum_gap_nm": delta_nm - 2 * coat_nm,
            "period_um": w_um + delta_nm / 1000.0,
            "temperature_K": temperature,
            "input_path": str(input_path),
            "result_dir": str(results_dir / name),
            "status": "prepared",
        })
    manifest = pd.DataFrame(records)
    manifest.to_csv(run_root / "manifest.csv", index=False)
    (run_root / "calculation_plan.json").write_text(json.dumps({
        "solver": "pss4_project local 1D periodic FED",
        "W_um": w_um,
        "delta_nm": delta_nm,
        "coating_nm": list(coatings),
        "coating_convention": "SiN(W/2)|SiO2(tc)|Vacuum(delta-2tc)|SiO2(tc)|SiN(W/2)",
        "temperature_K": temperature,
        "frequency_grid": "full material grid",
        "deliverables": ["hW", "spectra", "periodic_structure_schematic"],
    }, indent=2), encoding="utf-8")
    plot_schematic(run_root, w_um, delta_nm)
    return manifest


def _draw_periodic_stack(ax, w_um: float, delta_nm: float, coat_nm: float, title: str) -> None:
    layers = stack_layers(w_um, delta_nm, coat_nm)
    colors = {"SiN": "#5277A3", "SiO2": "#D89B5B", "Vacuum": "#E2E2E2"}
    total = sum(layer[1] for layer in layers)
    x = 0.0
    for material, thickness, _, _ in layers:
        ax.add_patch(Rectangle((x, 0), thickness, 1, facecolor=colors[material],
                               edgecolor="black", linewidth=0.8))
        if thickness / total > 0.07:
            ax.text(x + thickness / 2, 0.5, material, ha="center", va="center",
                    fontsize=9, rotation=90 if thickness / total < 0.16 else 0)
        x += thickness
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(total, color="black", linestyle="--", linewidth=1)
    ax.annotate("periodic boundary", (0, 1.04), ha="left", va="bottom", fontsize=8)
    ax.annotate("periodic boundary", (total, 1.04), ha="right", va="bottom", fontsize=8)
    ax.set_xlim(-0.02 * total, 1.02 * total)
    ax.set_ylim(-0.12, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("stack coordinate (µm)")
    ax.set_title(title)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)


def plot_schematic(run_root: Path, w_um: float, delta_nm: float) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.8), constrained_layout=True)
    _draw_periodic_stack(
        axes[0], w_um, delta_nm, 0,
        "No coating: SiN(W/2) | Vacuum(δ) | SiN(W/2)")
    endpoint = min(45.0, (delta_nm - 1.0) / 2.0)
    _draw_periodic_stack(
        axes[1], w_um, delta_nm, endpoint,
        f"Coated endpoint: SiN(W/2) | SiO$_2$({endpoint:g} nm) | Vacuum({delta_nm-2*endpoint:g} nm) | SiO$_2$({endpoint:g} nm) | SiN(W/2)")
    fig.suptitle(f"pss4-defined periodic unit cell: W={w_um:g} µm, δ={delta_nm:g} nm; Λ={w_um+delta_nm/1000:g} µm")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(run_root / f"pss4_periodic_structure_schematic.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def solver_command(input_path: Path, result_dir: Path, temperature: float,
                   freq_skip: int, k_points: int, resume: bool,
                   w_um: float, delta_nm: float, coat_nm: float) -> list[str]:
    command = [
        sys.executable, str(PROJECT_DIR / "main.py"),
        "--temperature", str(temperature), "--all-freq",
        "--freq-skip", str(freq_skip), "--k-grid-points", str(k_points),
        "--out-dir", str(result_dir),
        "--meta", f"W_um={w_um:g}", "--meta", f"delta_um={delta_nm / 1000.0:g}",
        "--meta", f"coating_thickness_um={coat_nm / 1000.0:g}",
    ]
    if resume:
        command.append("--resume")
    command.append(str(input_path))
    return command


def run_one(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=PROJECT_DIR, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, completed.stdout[-2000:]


def run_cases(run_root: Path, coatings: Sequence[int], temperature: float,
              freq_skip: int, k_points: int, workers: int, resume: bool,
              w_um: float, delta_nm: float) -> None:
    jobs = []
    for coat_nm in coatings:
        name = case_name(coat_nm)
        result_dir = run_root / "results" / name
        command = solver_command(run_root / "inputs" / f"{name}.txt", result_dir,
                                 temperature, freq_skip, k_points, resume,
                                 w_um, delta_nm, coat_nm)
        jobs.append((coat_nm, command, result_dir / "batch_stdout.log"))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_one, cmd, log): coat for coat, cmd, log in jobs}
        for future in concurrent.futures.as_completed(future_map):
            coat = future_map[future]
            code, tail = future.result()
            print(f"t_c={coat:02d} nm: {'complete' if code == 0 else 'FAILED'}", flush=True)
            if code != 0:
                failures.append((coat, tail))
    if failures:
        details = "\n".join(f"t_c={coat} nm:\n{tail}" for coat, tail in failures)
        raise RuntimeError(f"{len(failures)} pss4 cases failed:\n{details}")


def run_convergence(run_root: Path, temperature: float, workers: int,
                    w_um: float, delta_nm: float) -> int:
    """Run endpoint/midpoint convergence checks and return the final k-grid size."""
    pilot_coatings = (0, 22, 45)
    conv_root = run_root / "convergence"
    jobs = []
    # k convergence uses the full frequency grid so that its decision applies
    # directly to the production calculations.
    for coat_nm in pilot_coatings:
        for k_points in (500, 1000, 2000):
            result_dir = conv_root / f"k{k_points}" / case_name(coat_nm)
            command = solver_command(
                run_root / "inputs" / f"{case_name(coat_nm)}.txt", result_dir,
                temperature, 1, k_points, True, w_um, delta_nm, coat_nm)
            jobs.append((f"k={k_points}, tc={coat_nm}", command,
                         result_dir / "batch_stdout.log"))
    # Frequency convergence at the nominal 2000-point k grid. skip=1 is
    # already produced above and is reused during summarization.
    for coat_nm in pilot_coatings:
        for freq_skip in (15, 5):
            result_dir = conv_root / f"freqskip{freq_skip}_k2000" / case_name(coat_nm)
            command = solver_command(
                run_root / "inputs" / f"{case_name(coat_nm)}.txt", result_dir,
                temperature, freq_skip, 2000, True, w_um, delta_nm, coat_nm)
            jobs.append((f"freq_skip={freq_skip}, tc={coat_nm}", command,
                         result_dir / "batch_stdout.log"))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_one, cmd, log): label for label, cmd, log in jobs}
        for future in concurrent.futures.as_completed(future_map):
            label = future_map[future]
            code, tail = future.result()
            print(f"convergence {label}: {'complete' if code == 0 else 'FAILED'}", flush=True)
            if code != 0:
                failures.append((label, tail))
    if failures:
        details = "\n".join(f"{label}:\n{tail}" for label, tail in failures)
        raise RuntimeError(f"{len(failures)} convergence jobs failed:\n{details}")

    records = []
    for coat_nm in pilot_coatings:
        h_by_k = {}
        for k_points in (500, 1000, 2000):
            path = probe_result_path(conv_root / f"k{k_points}" / case_name(coat_nm))
            h_by_k[k_points] = integrated_hw(path)[0]
        for k_points in (500, 1000, 2000):
            records.append({"check": "k_grid", "coating_nm": coat_nm,
                            "setting": k_points, "hW_W_m2_K": h_by_k[k_points]})
        records[-1]["relative_change_from_previous"] = abs(
            h_by_k[2000] - h_by_k[1000]) / max(abs(h_by_k[2000]), 1e-300)
        for freq_skip in (15, 5, 1):
            if freq_skip == 1:
                path = probe_result_path(conv_root / "k2000" / case_name(coat_nm))
            else:
                path = probe_result_path(
                    conv_root / f"freqskip{freq_skip}_k2000" / case_name(coat_nm))
            records.append({"check": "frequency", "coating_nm": coat_nm,
                            "setting": freq_skip, "hW_W_m2_K": integrated_hw(path)[0]})
    table = pd.DataFrame(records)
    k2000 = table[(table.check == "k_grid") & (table.setting == 2000)]
    max_relative = float(k2000["relative_change_from_previous"].max())
    final_k = 4000 if max_relative > 0.01 else 2000
    table.to_csv(conv_root / "convergence_summary.csv", index=False)
    (conv_root / "convergence_decision.json").write_text(json.dumps({
        "criterion": "max abs(hW_2000-hW_1000)/abs(hW_2000) <= 0.01",
        "max_relative_change_1000_to_2000": max_relative,
        "selected_final_k_grid_points": final_k,
        "pilot_coating_nm": list(pilot_coatings),
        "frequency_skip_values": [15, 5, 1],
    }, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"Selected final k grid: {final_k}", flush=True)
    return final_k


def integrated_hw(csv_path: Path) -> tuple[float, pd.DataFrame]:
    frame = pd.read_csv(csv_path).sort_values(OMEGA_COLUMN)
    if SPECTRAL_COLUMN not in frame:
        raise ValueError(f"Missing {SPECTRAL_COLUMN} in {csv_path}")
    omega = frame[OMEGA_COLUMN].to_numpy(float)
    spectrum = frame[SPECTRAL_COLUMN].to_numpy(float)
    if len(frame) < 2 or not np.all(np.isfinite(spectrum)) or np.any(spectrum < -1e-20):
        raise ValueError(f"Invalid spectrum in {csv_path}")
    if hasattr(np, "trapezoid"):
        value = np.trapezoid(spectrum, omega)
    else:
        value = np.trapz(spectrum, omega)
    return float(value), frame


def probe_result_path(result_dir: Path) -> Path:
    """Return the single total-probe CSV regardless of the probe layer ID."""
    candidates = sorted(result_dir.glob("tot_P*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one tot_P*.csv in {result_dir}, found {len(candidates)}")
    return candidates[0]


def collect_results(run_root: Path, coatings: Sequence[int], expected_rows: int | None = None) -> pd.DataFrame:
    summaries = []
    long_frames = []
    wide = None
    for coat_nm in coatings:
        path = probe_result_path(run_root / "results" / case_name(coat_nm))
        hw, frame = integrated_hw(path)
        if expected_rows is not None and len(frame) != expected_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}")
        gap_nm = 100 - 2 * coat_nm
        summaries.append({"coating_nm": coat_nm, "effective_vacuum_gap_nm": gap_nm,
                          "hW_W_m2_K": hw, "frequency_rows": len(frame)})
        selected = frame[[OMEGA_COLUMN, SPECTRAL_COLUMN]].copy()
        selected.insert(0, "effective_vacuum_gap_nm", gap_nm)
        selected.insert(0, "coating_nm", coat_nm)
        long_frames.append(selected)
        column = f"coat_{coat_nm:03d}nm"
        series = frame[[OMEGA_COLUMN, SPECTRAL_COLUMN]].rename(columns={SPECTRAL_COLUMN: column})
        wide = series if wide is None else wide.merge(series, on=OMEGA_COLUMN, how="outer")
    summary = pd.DataFrame(summaries)
    summary.to_csv(run_root / "hW_by_coating.csv", index=False)
    pd.concat(long_frames, ignore_index=True).to_csv(run_root / "spectra_long.csv", index=False)
    wide.sort_values(OMEGA_COLUMN).to_csv(run_root / "spectra_wide.csv", index=False)
    plot_summary(run_root, summary, long_frames)
    manifest_path = run_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.drop(columns=[c for c in ("hW_W_m2_K", "frequency_rows") if c in manifest])
    manifest = manifest.merge(summary, on=["coating_nm", "effective_vacuum_gap_nm"], how="left")
    manifest["status"] = "complete"
    manifest.to_csv(manifest_path, index=False)
    return summary


def plot_summary(run_root: Path, summary: pd.DataFrame,
                 long_frames: Sequence[pd.DataFrame]) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(summary["coating_nm"], summary["hW_W_m2_K"], "o-", ms=3)
    ax.set(xlabel="SiO₂ coating thickness $t_c$ (nm)", ylabel="$h_W$ (W m$^{-2}$ K$^{-1}$)")
    ax.grid(True, alpha=0.25)
    save_figure(fig, run_root / "hW_vs_coating")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(summary["effective_vacuum_gap_nm"], summary["hW_W_m2_K"], "o-", ms=3)
    ax.set(xlabel="Effective vacuum gap $δ-2t_c$ (nm)", ylabel="$h_W$ (W m$^{-2}$ K$^{-1}$)")
    ax.grid(True, alpha=0.25)
    save_figure(fig, run_root / "hW_vs_effective_gap")
    representative = {0, 15, 30, 45}
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for frame in long_frames:
        coat = int(frame["coating_nm"].iloc[0])
        if coat in representative:
            ax.plot(frame[OMEGA_COLUMN], frame[SPECTRAL_COLUMN], label=f"{coat} nm")
    ax.set(xscale="log", yscale="log", xlabel="$ω$ (rad s$^{-1}$)",
           ylabel="Spectral $h_W$ (W m$^{-2}$ K$^{-1}$ / (rad s$^{-1}$))")
    ax.legend(title="$t_c$")
    ax.grid(True, which="both", alpha=0.2)
    save_figure(fig, run_root / "representative_spectra")

    plot_spectral_flux_colorbar(run_root, long_frames)


def plot_spectral_flux_colorbar(run_root: Path,
                                long_frames: Sequence[pd.DataFrame],
                                basename: str = "spectral_flux_vs_frequency_coating_colorbar") -> None:
    """Plot every coating spectrum with coating thickness encoded by a colorbar."""
    coatings = np.array([float(frame["coating_nm"].iloc[0]) for frame in long_frames])
    norm = plt.Normalize(vmin=float(coatings.min()), vmax=float(coatings.max()))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for frame, coat_nm in zip(long_frames, coatings):
        omega = frame[OMEGA_COLUMN].to_numpy(float)
        flux = frame[SPECTRAL_COLUMN].to_numpy(float)
        valid = np.isfinite(omega) & np.isfinite(flux) & (omega > 0) & (flux > 0)
        ax.plot(omega[valid], flux[valid], color=cmap(norm(coat_nm)),
                linewidth=0.8, alpha=0.85)
    ax.set(xscale="log", yscale="log", xlabel="$ω$ (rad s$^{-1}$)",
           ylabel="Spectral flux (W m$^{-2}$ K$^{-1}$ / (rad s$^{-1}$))")
    ax.grid(True, which="both", alpha=0.18)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.025)
    colorbar.set_label("SiO$_2$ coating thickness $t_c$ (nm)")
    colorbar.set_ticks(np.arange(math.ceil(norm.vmin / 5) * 5, norm.vmax + 0.1, 5))
    save_figure(fig, run_root / basename)


def save_figure(fig, base_path: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(base_path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--W-um", type=float, default=1.0)
    parser.add_argument("--delta-nm", type=int, default=100)
    parser.add_argument("--coat-start-nm", type=int, default=0)
    parser.add_argument("--coat-stop-nm", type=int, default=45)
    parser.add_argument("--coat-step-nm", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freq-skip", type=int, default=1)
    parser.add_argument("--k-grid-points", type=int, default=2000)
    parser.add_argument("--expected-rows", type=int, default=2502)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--convergence-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coatings = coating_values(args.coat_start_nm, args.coat_stop_nm, args.coat_step_nm)
    run_root = make_run_root(args.output_root)
    print(f"Output root: {run_root}", flush=True)
    if not args.collect_only:
        prepare_inputs(run_root, args.W_um, args.delta_nm, coatings, args.temperature)
    if args.prepare_only:
        return 0
    if args.convergence_only:
        run_convergence(run_root, args.temperature, args.workers, args.W_um, args.delta_nm)
        return 0
    if not args.collect_only:
        run_cases(run_root, coatings, args.temperature, args.freq_skip,
                  args.k_grid_points, args.workers, args.resume,
                  args.W_um, args.delta_nm)
    expected = args.expected_rows if args.freq_skip == 1 else None
    summary = collect_results(run_root, coatings, expected)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
