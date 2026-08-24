# Agent Instructions

## Standard Heatmap Workflow

For tau heatmap generation, use the established analysis scripts as the
standard method:

- Batch generation: `D:\Researches\wqs_comps\scripts\batch_visualize_tau_heatmaps.py`
- Single-result generation: `D:\Researches\wqs_comps\scripts\visualize_tau_heatmaps.py`

Do not use ad hoc local heatmap helpers as the default when reproducing or
extending existing heatmap outputs. Match the existing analysis convention:

```powershell
python D:\Researches\wqs_comps\scripts\batch_visualize_tau_heatmaps.py --root <sweep-root> --output-dir <analysis-output-root> --combo-index <indices> --pol both --y-axis hbaromega --cmap jet --overlay-refinement-samples
```

For outputs like
`hw_k_heatmaps_combo00_01_refine2_ksub1_refined_raster_blankfill_sep_hbaromega_jet_*`,
the expected producer is `batch_visualize_tau_heatmaps.py`, using solver result
directories that contain `P*_tau_s.csv`, `P*_tau_p.csv`, `k_parallel_grid.csv`,
and optional `P*_tau_{s,p}_zero_refinement_samples.csv`.

When a prop/evan division curve is requested, use
`--draw-prop-evan-boundary`. The standard material boundary is computed as
`k_parallel = Re(sqrt(epsilon)) * omega / c` from the selected dielectric
function file, and should be drawn as a solid white curve at the highest
overlay z-order so refinement rasters or zero-line overlays cannot cover it.

Preserve the established output layout:

```text
<analysis-output-root>/
  heatmap_manifest.csv
  coarse_sweep/combo_*/results_coat_*/*.png
  fine_sweep/combo_*/results_coat_*/*.png
```

## Figure Typography

Apply these Matplotlib theme settings to future figure generation unless the
user explicitly asks otherwise:

- Main font family: Arial.
- Axis titles/labels and colorbar labels: 18 pt.
- Tick labels and colorbar tick labels: 16 pt.
- Greek math symbols, including `\tau` and `\omega`, must render in Times New
  Roman; Latin text, units, and non-Greek math roman text should remain Arial.
- Use proper math notation in labels:
  - `\hbar\omega` instead of `hbar*omega`.
  - `\mathrm{k}_{\parallel}\;(\mathrm{m}^{-1})` instead of `k//(1/m)`.
