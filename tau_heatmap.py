import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

DEFAULT_BOUNDARY_STYLE = {
    "color": "white",
    "linestyle": "--",
    "linewidth": 1.0,
}


def plot_tau_heatmap(tau_grid_result: Dict, polarization: str = "p",
                     draw_prop_evan_boundary: bool = True,
                     boundary_style: Optional[Dict] = None,
                     output_path: Optional[str] = None,
                     cmap: str = "viridis",
                     log_scale: bool = True,
                     title: Optional[str] = None,
                     ax=None) -> Tuple:
    """Plot a tau(omega, k_parallel) heatmap from compute_tau_grid output."""
    omega = tau_grid_result["omega"]
    k_parallel = tau_grid_result["k_parallel"]
    tau_by_pol = tau_grid_result["tau"]

    if polarization not in tau_by_pol:
        raise ValueError(f"Polarization '{polarization}' not found in tau grid")

    tau = np.asarray(tau_by_pol[polarization])
    if tau.shape != (len(omega), len(k_parallel)):
        raise ValueError(
            f"Expected tau shape {(len(omega), len(k_parallel))}, got {tau.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    mesh_data = tau.T
    norm = None
    if log_scale:
        positive = mesh_data[mesh_data > 0]
        if positive.size:
            mesh_data = np.ma.masked_less_equal(mesh_data, 0)
            norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))

    mesh = ax.pcolormesh(omega, k_parallel, mesh_data, shading="auto",
                         cmap=cmap, norm=norm)
    fig.colorbar(mesh, ax=ax, label=f"tau_{polarization}")

    if draw_prop_evan_boundary:
        style = dict(DEFAULT_BOUNDARY_STYLE)
        if boundary_style:
            style.update(boundary_style)
        ax.plot(omega, tau_grid_result["prop_evan_boundary"], **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("omega (rad/s)")
    ax.set_ylabel("k_parallel (1/m)")
    if title:
        ax.set_title(title)
    else:
        probe_id = tau_grid_result.get("probe_layer_id", "?")
        ax.set_title(f"tau_{polarization}(omega, k_parallel), probe P{probe_id}")

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax, mesh
