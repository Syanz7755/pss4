#!/usr/bin/env python
"""
Compute heff vs. vertical offset d for W=0.6667 um, H=10 um (H/W=15).

Two methods (per 04_manuscript_theory_model.md):
  1. Direct:  heff_direct = hW * (H - d) / (W + delta)
  2. HT model: non-isothermal slab model with
       beta  = sqrt(4 * hW / (k_SiN * W))
       L_th  = 1 / beta
       Solve theta'' = beta^2 * theta in the overlap region,
       then extract device-level heff from total heat flow Q
       normalized by A_FP = 2 * W * (H + d).

The hW(T) values are read from the existing combined CSV
(combined_row_level_heff_inputs_loglog_wstyle.csv) which contains
multi-temperature data for W=0.6667, AR15 at all 6 temperatures.

Usage:
    python postprocess_heff_vs_d.py
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ──────────────────────────────────────────────
c = 299792458.0
hbar = 1.0545718e-34
k_B = 1.380649e-23

# ── Geometry parameters ────────────────────────────────────────────
W_um = 0.6667
H_um = 10.0
delta_um = 0.6  # vacuum gap before coating
W_m = W_um * 1e-6
H_m = H_um * 1e-6
delta_m = delta_um * 1e-6

# k_SiN from Zink & Hellman (2003) / Ftouni et al. (2015)
# Consistent with existing batch data (reverse-engineered ~4.28 W/(m·K))
k_SiN = 4.28  # W/(m·K)

# Temperature sweep
temps_K = np.array([70, 100, 150, 200, 250, 300])

# d sweep: vertical offset between slab roots, um
d_um_values = np.linspace(0.5, H_um - 0.5, 19)  # 0.5 to 9.5 um

# ── Output directory ───────────────────────────────────────────────
out_dir = "heff_vs_d_W0p6667_H10_20260709"
os.makedirs(out_dir, exist_ok=True)

# ── Read hW(T) from existing data ──────────────────────────────────
csv_file = os.path.join(
    "combined_delta0p600_HW3to100_heff_temperature_plots_20260707_corrected_semilogy",
    "combined_row_level_heff_inputs_loglog_wstyle.csv"
)
df = pd.read_csv(csv_file)

# Filter for W=0.6667, AR=15
mask_W = df["W_um"] == 0.6667
mask_AR = df["aspect_ratio"] == 15.0
df_AR15 = df[mask_W & mask_AR].copy()

print(f"Found {len(df_AR15)} rows for W=0.6667, AR=15")
print(f"Configurations: {df_AR15['cfg_id'].unique()}")

# Extract hW for each temperature and config
# Two configs: gap600_coat000 (no_coat) and gap200_coat200 (coat_200nm)
configs = {}
for cfg_id in df_AR15["cfg_id"].unique():
    row = df_AR15[df_AR15["cfg_id"] == cfg_id].iloc[0]
    cfg_info = {
        "cfg_id": cfg_id,
        "coat_label": row["coat_label"],
        "d_eff_um": row["d_eff_um"],
        "tot_file": row["tot_file"],
    }
    # Get hW at each temperature
    hW_by_T = {}
    for T in temps_K:
        trow = df_AR15[(df_AR15["cfg_id"] == cfg_id) & (df_AR15["temperature_K"] == T)]
        if len(trow) > 0:
            hW_by_T[T] = trow["hW_W_m2_K"].values[0]
        else:
            hW_by_T[T] = None
    cfg_info["hW_by_T"] = hW_by_T
    configs[cfg_id] = cfg_info

# ── HT model solver ────────────────────────────────────────────────
# Per 04_manuscript_theory_model.md, Section "3. Non-Isothermal Slab Model"
#
# Regions:
#   1: 0 <= y <= d          (hot slab only)
#   2: d <= y <= H          (overlap, both slabs coupled by hW)
#   3: H <= y <= H+d        (cold slab only)
#
# Boundary conditions:
#   T_H(0) = T_H0
#   T_C(H+d) = T_C0
#   dT_H/dy(d) = 0  (adiabatic at root of hot slab into overlap)
#   dT_C/dy(H) = 0  (adiabatic at tip of cold slab into non-overlap)
#
# For theta = T_H - T_C in overlap:
#   theta'' = beta^2 * theta
#   beta = sqrt(4*hW / (k_SiN * W))
#   L_th = 1/beta
#
# Solution (derived in SI):
#   Let L = H - d  (overlap length)
#   gamma = beta * L
#   theta(y) = T0 * cosh(beta * (2*H + d - y)) / cosh(gamma * (1 + d/(2*L)))
#     where T0 = theta(d) = T_H(d) - T_C(d)
#
# Total heat flow Q (from conductive flux at hot base):
#   Q = -k_SiN * W * dT_H/dy(0)
#     = k_SiN * W * T0 / L_th * tanh(gamma * (1 + d/(2*L))) / (d / L_th)
#     = k_SiN * W / L_th * T0 * tanh(gamma * (1 + d/(2*L))) / (d / L_th)
#     = k_SiN * W / d * T0 * tanh(gamma * (1 + d/(2*L)))
#
# Global temperature difference:
#   Delta_T_global = T_H0 - T_C0
#   With T_H0 - T_C0 = T0 / cosh(gamma * (1 + d/(2*L)))   (when d>0)
#
# Effective coefficient:
#   heff = Q / (A_FP * Delta_T_global)
#   A_FP = 2 * W * (H + d)

def compute_heff_HT(hW, d_um, H_um, W_um, delta_um, k_SiN):
    """
    Compute heff from the non-isothermal slab model.
    
    Notation follows 04_manuscript_theory_model.md, Section 3.
    
    The model solves theta'' = beta^2 * theta in the overlap region,
    where beta = sqrt(4*hW / (k_SiN * W)) and L_th = 1/beta.
    
    The device-level heff is normalized by the same area as the direct
    method: A = (W + delta), per unit overlap length.
    
    For d = 0:
        heff = hW * tanh(H/L_th) / (H/L_th) * H / (W + delta)
    
    For d > 0:
        Overlap length L = H - d.
        heff = hW * tanh(L/L_th) / (L/L_th) * L / (W + delta)
    """
    L = H_um - d_um  # overlap length, um
    if L <= 0:
        return 0.0
    
    beta = np.sqrt(4.0 * hW / (k_SiN * W_um * 1e-6))  # 1/m
    L_th = 1.0 / beta  # m
    L_th_um = L_th * 1e6  # um
    
    gamma = L / L_th_um  # dimensionless (both in um)
    if gamma < 1e-12:
        ht_correction = 1.0
    else:
        ht_correction = np.tanh(gamma) / gamma
    
    heff = hW * ht_correction * L / (W_um + delta_um)
    
    return heff


def compute_heff_direct(hW, d_um, H_um, W_um, delta_um):
    """
    Direct geometric estimate: heff_direct = hW * (H - d) / (W + delta)
    
    This is the exploratory scaling formula (per 04_manuscript_theory_model.md).
    """
    L = H_um - d_um  # overlap length
    if L <= 0:
        return 0.0
    return hW * L / (W_um + delta_um)


def compute_L_th(hW, W_um, k_SiN):
    """Thermal penetration length in um."""
    L_th_m = np.sqrt(k_SiN * W_um * 1e-6 / (4.0 * hW))
    return L_th_m * 1e6  # um


# ── Compute heff for each config, temperature, and d ───────────────
all_rows = []

for cfg_id, cfg in configs.items():
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is None:
            continue
        
        L_th = compute_L_th(hW, W_um, k_SiN)
        
        for d_val in d_um_values:
            heff_d = compute_heff_direct(hW, d_val, H_um, W_um, delta_um)
            heff_ht = compute_heff_HT(hW, d_val, H_um, W_um, delta_um, k_SiN)
            
            all_rows.append({
                "temperature_K": T,
                "cfg_id": cfg_id,
                "coat_label": cfg["coat_label"],
                "W_um": W_um,
                "H_um": H_um,
                "delta_um": delta_um,
                "d_eff_um": cfg["d_eff_um"],
                "d_um": d_val,
                "overlap_um": H_um - d_val,
                "hW_W_m2_K": hW,
                "L_th_um": L_th,
                "heff_direct_W_m2_K": heff_d,
                "heff_HT_W_m2_K": heff_ht,
            })

df_result = pd.DataFrame(all_rows)
print(f"\nTotal data points: {len(df_result)}")

# ── Verify against existing data at d=0 ────────────────────────────
print("\n=== Verification against existing data (d=0, from CSV) ===")
for cfg_id, cfg in configs.items():
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is None:
            continue
        # From existing CSV
        existing = df_AR15[
            (df_AR15["cfg_id"] == cfg_id) & (df_AR15["temperature_K"] == T)
        ]
        if len(existing) == 0:
            continue
        existing_heff_d = existing["heff_direct_W_m2_K"].values[0]
        existing_heff_ht = existing["heff_HT_W_m2_K"].values[0]
        
        # Our computation at d=0
        our_heff_d = compute_heff_direct(hW, 0.0, H_um, W_um, delta_um)
        our_heff_ht = compute_heff_HT(hW, 0.0, H_um, W_um, delta_um, k_SiN)
        
        print(f"  T={T}K, {cfg['coat_label']}: "
              f"hW={hW:.4f}")
        print(f"    Direct: existing={existing_heff_d:.6f}, ours={our_heff_d:.6f}, "
              f"diff={abs(existing_heff_d - our_heff_d)/existing_heff_d*100:.2f}%")
        print(f"    HT:     existing={existing_heff_ht:.6f}, ours={our_heff_ht:.6f}, "
              f"diff={abs(existing_heff_ht - our_heff_ht)/existing_heff_ht*100:.2f}%")

# ── Power-law fitting: y = A * x^B ────────────────────────────────
def power_law(x, A, B):
    return A * x**B

print("\n=== Power-law fitting: heff = A * d^B ===")
fit_results = {}

for cfg_id, cfg in configs.items():
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is None:
            continue
        
        subset = df_result[
            (df_result["cfg_id"] == cfg_id) & 
            (df_result["temperature_K"] == T)
        ]
        d = subset["d_um"].values
        heff_d = subset["heff_direct_W_m2_K"].values
        heff_ht = subset["heff_HT_W_m2_K"].values
        
        # Fit direct method
        try:
            popt_d, _ = curve_fit(power_law, d, heff_d, p0=[1, 0])
            A_d, B_d = popt_d
            r2_d = 1 - np.sum((heff_d - power_law(d, A_d, B_d))**2) / np.sum((heff_d - np.mean(heff_d))**2)
        except:
            A_d, B_d, r2_d = np.nan, np.nan, np.nan
        
        # Fit HT model
        try:
            popt_ht, _ = curve_fit(power_law, d, heff_ht, p0=[1, 0])
            A_ht, B_ht = popt_ht
            r2_ht = 1 - np.sum((heff_ht - power_law(d, A_ht, B_ht))**2) / np.sum((heff_ht - np.mean(heff_ht))**2)
        except:
            A_ht, B_ht, r2_ht = np.nan, np.nan, np.nan
        
        fit_results[(cfg_id, T)] = {
            "A_direct": A_d, "B_direct": B_d, "R2_direct": r2_d,
            "A_HT": A_ht, "B_HT": B_ht, "R2_HT": r2_ht,
        }
        
        print(f"  T={T}K, {cfg['coat_label']}, hW={hW:.4f}:")
        print(f"    Direct: heff = {A_d:.4e} * d^{B_d:.4f}  (R²={r2_d:.6f})")
        print(f"    HT:     heff = {A_ht:.4e} * d^{B_ht:.4f}  (R²={r2_ht:.6f})")

# ── Save xlsx ──────────────────────────────────────────────────────
xlsx_path = os.path.join(out_dir, "heff_vs_d_W0p6667_H10.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    # Main data
    df_result.to_excel(writer, sheet_name="heff_vs_d", index=False)
    
    # Fit results
    fit_rows = []
    for (cfg_id, T), fr in fit_results.items():
        fit_rows.append({
            "temperature_K": T,
            "cfg_id": cfg_id,
            "A_direct": fr["A_direct"],
            "B_direct": fr["B_direct"],
            "R2_direct": fr["R2_direct"],
            "A_HT": fr["A_HT"],
            "B_HT": fr["B_HT"],
            "R2_HT": fr["R2_HT"],
        })
    pd.DataFrame(fit_rows).to_excel(writer, sheet_name="fit_results", index=False)
    
    # Summary table: hW and L_th by temperature
    summary_rows = []
    for cfg_id, cfg in configs.items():
        for T in temps_K:
            hW = cfg["hW_by_T"][T]
            if hW is None:
                continue
            L_th = compute_L_th(hW, W_um, k_SiN)
            summary_rows.append({
                "temperature_K": T,
                "cfg_id": cfg_id,
                "coat_label": cfg["coat_label"],
                "hW_W_m2_K": hW,
                "L_th_um": L_th,
            })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="hW_by_temperature", index=False)

print(f"\nSaved xlsx to {xlsx_path}")

# ── Save CSV ───────────────────────────────────────────────────────
csv_out = os.path.join(out_dir, "heff_vs_d_W0p6667_H10.csv")
df_result.to_csv(csv_out, index=False)
print(f"Saved CSV to {csv_out}")

# ── Plots ──────────────────────────────────────────────────────────
# Use same style as existing combined plots (loglog, with style)

# Color map for temperatures
cmap = plt.cm.viridis
temp_colors = {T: cmap(i / (len(temps_K) - 1)) for i, T in enumerate(temps_K)}

# Marker styles
markers = {"no_coat": "o", "coat_200nm": "s"}

# ── Figure 1: heff_HT vs d, separated by config ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, cfg_id in enumerate(configs):
    ax = axes[ax_idx]
    cfg = configs[cfg_id]
    
    for T in temps_K:
        subset = df_result[
            (df_result["cfg_id"] == cfg_id) & 
            (df_result["temperature_K"] == T)
        ]
        d = subset["d_um"].values
        heff_ht = subset["heff_HT_W_m2_K"].values
        
        ax.loglog(d, heff_ht, "o-", 
                  label=f"{T}K", color=temp_colors[T],
                  markersize=5, linewidth=1.5,
                  markeredgecolor="white", markeredgewidth=0.5)
        
        # Add power-law fit curve
        fr = fit_results.get((cfg_id, T), {})
        if not np.isnan(fr.get("A_HT", np.nan)):
            d_fit = np.linspace(0.5, H_um - 0.5, 200)
            ax.loglog(d_fit, power_law(d_fit, fr["A_HT"], fr["B_HT"]),
                      "--", color=temp_colors[T], alpha=0.5, linewidth=1)
    
    ax.set_xlabel(r"$d$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$h_{\mathrm{eff}}^{\mathrm{HT}}$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
    ax.set_title(f"{cfg['coat_label']}\nW={W_um} $\mu$m, H={H_um} $\mu$m, "
                 f"$\delta$={delta_um} $\mu$m", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.4, 11)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "heff_HT_vs_d_loglog.png"), dpi=150)
plt.close()
print("Saved heff_HT_vs_d_loglog.png")

# ── Figure 2: heff_direct vs d, separated by config ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, cfg_id in enumerate(configs):
    ax = axes[ax_idx]
    cfg = configs[cfg_id]
    
    for T in temps_K:
        subset = df_result[
            (df_result["cfg_id"] == cfg_id) & 
            (df_result["temperature_K"] == T)
        ]
        d = subset["d_um"].values
        heff_d = subset["heff_direct_W_m2_K"].values
        
        ax.loglog(d, heff_d, "o-", 
                  label=f"{T}K", color=temp_colors[T],
                  markersize=5, linewidth=1.5,
                  markeredgecolor="white", markeredgewidth=0.5)
        
        # Add power-law fit curve
        fr = fit_results.get((cfg_id, T), {})
        if not np.isnan(fr.get("A_direct", np.nan)):
            d_fit = np.linspace(0.5, H_um - 0.5, 200)
            ax.loglog(d_fit, power_law(d_fit, fr["A_direct"], fr["B_direct"]),
                      "--", color=temp_colors[T], alpha=0.5, linewidth=1)
    
    ax.set_xlabel(r"$d$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$h_{\mathrm{eff}}^{\mathrm{direct}}$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
    ax.set_title(f"{cfg['coat_label']}\nW={W_um} $\mu$m, H={H_um} $\mu$m, "
                 f"$\delta$={delta_um} $\mu$m", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.4, 11)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "heff_direct_vs_d_loglog.png"), dpi=150)
plt.close()
print("Saved heff_direct_vs_d_loglog.png")

# ── Figure 3: heff_HT vs d, both configs together ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, cfg_id in enumerate(configs):
    ax = axes[ax_idx]
    cfg = configs[cfg_id]
    marker = markers.get(cfg["coat_label"], "o")
    
    for T in temps_K:
        subset = df_result[
            (df_result["cfg_id"] == cfg_id) & 
            (df_result["temperature_K"] == T)
        ]
        d = subset["d_um"].values
        heff_ht = subset["heff_HT_W_m2_K"].values
        
        ax.loglog(d, heff_ht, f"{marker}-", 
                  label=f"{T}K (hW={subset['hW_W_m2_K'].values[0]:.2f})",
                  color=temp_colors[T],
                  markersize=5, linewidth=1.5,
                  markeredgecolor="white", markeredgewidth=0.5)
    
    ax.set_xlabel(r"$d$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$h_{\mathrm{eff}}^{\mathrm{HT}}$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
    ax.set_title(f"{cfg['coat_label']}", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.4, 11)

plt.suptitle(f"W={W_um} um, H={H_um} um (H/W=15), "
             f"delta={delta_um} um, k_SiN={k_SiN} W/(m*K)",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "heff_HT_vs_d_combined.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved heff_HT_vs_d_combined.png")

# ── Figure 4: hW vs temperature ───────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for cfg_id, cfg in configs.items():
    marker = markers.get(cfg["coat_label"], "o")
    T_vals = []
    hW_vals = []
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is not None:
            T_vals.append(T)
            hW_vals.append(hW)
    ax.loglog(T_vals, hW_vals, f"{marker}-", 
              label=cfg["coat_label"], markersize=6, linewidth=2)

ax.set_xlabel(r"Temperature (K)", fontsize=12)
ax.set_ylabel(r"$h_W$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
ax.set_title(f"W={W_um} $\mu$m, H={H_um} $\mu$m", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(60, 350)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "hW_vs_temperature.png"), dpi=150)
plt.close()
print("Saved hW_vs_temperature.png")

# ── Figure 5: L_th vs temperature ─────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for cfg_id, cfg in configs.items():
    marker = markers.get(cfg["coat_label"], "o")
    T_vals = []
    Lth_vals = []
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is not None:
            T_vals.append(T)
            Lth_vals.append(compute_L_th(hW, W_um, k_SiN))
    ax.semilogy(T_vals, Lth_vals, f"{marker}-", 
                label=cfg["coat_label"], markersize=6, linewidth=2)

ax.set_xlabel(r"Temperature (K)", fontsize=12)
ax.set_ylabel(r"$L_{\mathrm{th}}$ ($\mu$m)", fontsize=12)
ax.set_title(f"W={W_um} $\mu$m, k_SiN={k_SiN} W/(m$\cdot$K)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(60, 350)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "Lth_vs_temperature.png"), dpi=150)
plt.close()
print("Saved Lth_vs_temperature.png")

# ── Figure 6: Comparison of direct vs HT model at d=0 ─────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for cfg_id, cfg in configs.items():
    marker = markers.get(cfg["coat_label"], "o")
    T_vals = []
    heff_d_vals = []
    heff_ht_vals = []
    for T in temps_K:
        hW = cfg["hW_by_T"][T]
        if hW is not None:
            T_vals.append(T)
            heff_d_vals.append(compute_heff_direct(hW, 0.0, H_um, W_um, delta_um))
            heff_ht_vals.append(compute_heff_HT(hW, 0.0, H_um, W_um, delta_um, k_SiN))
    ax.loglog(T_vals, heff_d_vals, f"{marker}", 
              label=f"{cfg['coat_label']} (direct)", markersize=6, alpha=0.7)
    ax.loglog(T_vals, heff_ht_vals, f"{marker}", 
              label=f"{cfg['coat_label']} (HT)", markersize=6,
              markerfacecolor="none", markeredgecolor=temp_colors[300],
              markeredgewidth=2)

ax.set_xlabel(r"Temperature (K)", fontsize=12)
ax.set_ylabel(r"$h_{\mathrm{eff}}$ (W m$^{-2}$ K$^{-1}$)", fontsize=12)
ax.set_title(f"W={W_um} $\mu$m, H={H_um} $\mu$m, d=0", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(60, 350)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "heff_comparison_d0.png"), dpi=150)
plt.close()
print("Saved heff_comparison_d0.png")

# ── Save CSV of heff vs d for easy post-processing ────────────────
for method in ["heff_direct_W_m2_K", "heff_HT_W_m2_K"]:
    short_name = method.replace("_W_m2_K", "")
    csv_method = os.path.join(out_dir, f"heff_vs_d_{short_name}.csv")
    df_result[["temperature_K", "cfg_id", "coat_label", "d_um", 
               "overlap_um", "hW_W_m2_K", method]].to_csv(csv_method, index=False)

print(f"\nAll outputs saved to {out_dir}/")
print("Done.")
