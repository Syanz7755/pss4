import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config_parser import parse_stack_file
from grid_builder import build_k_grid
from main import parse_float_file, parse_int_items
from materials import load_materials_and_grid
from tau_grid import compute_tau_grid
from tau_heatmap import plot_tau_heatmap


def _find_probe_index(parsed_layers, probe_layer_id):
    probe_indices = [i for i, layer in enumerate(parsed_layers) if layer["is_probe"]]
    if not probe_indices:
        raise ValueError("No probe layer found in input stack")
    if probe_layer_id is None:
        return probe_indices[0]
    for idx in probe_indices:
        if parsed_layers[idx]["layer_id"] == probe_layer_id:
            return idx
    raise ValueError(f"Probe layer id {probe_layer_id} was not found")


def _find_gap_thickness(parsed_layers):
    for layer in parsed_layers:
        if layer["material"] == "Vacuum":
            return layer["thickness"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate tau(omega,k_parallel) heatmaps")
    parser.add_argument("input_path", type=str, help="Path to input stack file")
    parser.add_argument("--polarization", choices=["s", "p"], default="p")
    parser.add_argument("--probe-layer-id", type=int, default=None,
                        help="Probe layer id; defaults to the first probe layer")
    parser.add_argument("--all-freq", action="store_true", default=False,
                        help="Use full frequency grid")
    parser.add_argument("--freq-skip", type=int, default=1,
                        help="Use every Nth frequency from the selected material table")
    parser.add_argument("--freq-offset", type=int, default=0,
                        help="Starting offset for --freq-skip subsets")
    parser.add_argument("--freq-indices", type=str, default=None,
                        help="Comma-separated indices or a text file of indices")
    parser.add_argument("--freq-values-file", type=str, default=None,
                        help="Text file of explicit omega values in rad/s")
    parser.add_argument("--out-dir", type=str, default="results/",
                        help="Output directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path; defaults under --out-dir")
    parser.add_argument("--no-prop-evan-boundary", action="store_true", default=False,
                        help="Do not draw the propagating/evanescent boundary")
    parser.add_argument("--boundary-color", type=str, default="white")
    parser.add_argument("--boundary-linestyle", type=str, default="--")
    parser.add_argument("--boundary-linewidth", type=float, default=1.0)
    args = parser.parse_args()

    parsed_layers, _ = parse_stack_file(args.input_path)
    omega_grid, material_eps = load_materials_and_grid(
        parsed_layers,
        args.all_freq,
        freq_skip=args.freq_skip,
        freq_offset=args.freq_offset,
        freq_indices=parse_int_items(args.freq_indices),
        freq_values=parse_float_file(args.freq_values_file),
    )
    k_par_grid, _ = build_k_grid(omega_grid[0], _find_gap_thickness(parsed_layers))
    probe_idx = _find_probe_index(parsed_layers, args.probe_layer_id)

    tau_result = compute_tau_grid(
        parsed_layers,
        probe_idx,
        omega_grid,
        material_eps,
        k_par_grid,
        polarizations=(args.polarization,),
    )

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.out_dir, f"tau_{args.polarization}_P{tau_result['probe_layer_id']}_heatmap.png")

    boundary_style = {
        "color": args.boundary_color,
        "linestyle": args.boundary_linestyle,
        "linewidth": args.boundary_linewidth,
    }
    fig, _, _ = plot_tau_heatmap(
        tau_result,
        polarization=args.polarization,
        draw_prop_evan_boundary=not args.no_prop_evan_boundary,
        boundary_style=boundary_style,
        output_path=output_path,
    )
    fig.clf()
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
