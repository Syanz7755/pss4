# Tau Prop–Evan Post-processing

`scripts/analyze_tau_prop_evan.py` post-processes an existing PSS4 result that
contains `P{id}_tau_s.csv`, `P{id}_tau_p.csv`, and `k_parallel_grid.csv`. It
does not recompute the electromagnetic solution.

For a gap medium with complex relative dielectric function
`epsilon_gap(omega)`, the division is

```text
k_boundary(omega) = Re(sqrt(epsilon_gap(omega))) * omega / c.
```

Samples with `k_parallel < k_boundary` are propagating; samples at the
boundary and above it are evanescent. The dielectric table must have the PSS4
three-column format:

```text
omega(rad/s)    epsilon_real    epsilon_imag
```

For each polarization `mu` and regime `r` (`prop` or `evan`), the script uses
the saved quadrature weights to calculate

```text
Phi_mu,r(omega) = sum_{k in r} tau_mu(omega, k) * k * w_k / (2*pi),
Phi_r(omega) = Phi_s,r(omega) + Phi_p,r(omega),
h_omega,r = (dTheta/dT) * Phi_r(omega) / (2*pi).
```

Consequently, `total = prop + evan` at every frequency, apart from floating
point round-off. `h_omega` is the spectral flux / spectral heat-transfer
coefficient; integrating it over omega gives each regime's contribution to the
integrated local coefficient.

## Command

From `pss4_project/`, for the latest saved tau pair in this workspace:

```powershell
python scripts/analyze_tau_prop_evan.py `
  D:/Researches/wqs_comps/analysis/combo999_W1_AR30_delta200_tc0to50nm_n10_densefreq2_20260716/combo_999_09 `
  --probe 2 --temperature 300 `
  --epsilon-file materials/aligned_Vacuum.txt `
  --output-dir D:/Researches/wqs_comps/analysis/combo999_W1_AR30_delta200_tc0to50nm_n10_densefreq2_20260716/combo_999_09/prop_evan_analysis
```

The default dielectric table is `materials/aligned_Vacuum.txt`, so the
`--epsilon-file` option can be omitted for a vacuum gap. Supply the appropriate
gap-medium table to define a different material boundary.

## Periodic-structure (Bloch) classification

The gap light line above classifies channels in the probe/gap medium; it does
not classify a mode of the periodic material stack. To classify the exact unit
cell defined by a PSS4 input, use `scripts/analyze_tau_bloch_prop_evan.py` with
that same input file. It obtains the unit-cell transfer-matrix eigenvalue
`lambda = exp(i K_B Lambda)` separately for s and p. The forward/decaying root
is selected, and the mode is labelled Bloch-propagating when

```text
abs(Im(K_B)) <= eta * max(abs(Re(K_B)), omega / c),
```

with `eta=0.01` by default; otherwise it is labelled Bloch-evanescent. The
threshold is part of the output metadata because a lossy periodic stack has no
mathematically sharp prop/evan boundary.

```powershell
python scripts/analyze_tau_bloch_prop_evan.py <tau-result-dir> <exact-input-stack.txt> `
  --probe 2 --temperature 300 --eta 0.01 --output-dir <bloch-output-dir>
```

This writes s/p tau diagrams with a white dashed contour and a translucent
Bloch-evanescent overlay, along with `bloch_prop_evan_observables.csv` and
the associated transmission/spectral-flux figures. Keep these results separate
from the gap-light-line outputs: the two decompositions have different physical
meanings and are complementary.

## Outputs

- `tau_s_bwr_prop_evan.png`, `tau_p_bwr_prop_evan.png`: red–white–blue,
  log-scaled tau maps with the material boundary as a white dashed curve.
- `transmission_prop_evan.png`, `spectral_flux_prop_evan.png`: total,
  propagating, and evanescent curves.
- `prop_evan_observables.csv`: every frequency's total/prop/evan values,
  including separate s and p transmission contributions.
- `prop_evan_boundary.csv`: epsilon and the calculated boundary used at every
  saved omega.
- `raw_input_csv/`: exact copies of tau_s, tau_p, k-grid, and dielectric input
  CSV/text files used to make the analysis reproducible.
