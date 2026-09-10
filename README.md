# pss4 — 1D Periodic Fluctuational Electrodynamics Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

pss4 computes near-field radiative heat transfer coefficients (RHTC) for 1D periodic planar structures using the S-matrix (Redheffer Star Product) method. It supports multi-layer unit cells with arbitrary material stacking under periodic boundary conditions.

## Recommended Solver Tree

Use `pss4_project/` for new PSS4 calculations. The sibling workspace directory
`pss4/` is deprecated and should be treated as a legacy/experimental copy for
historical scripts and diagnostics only.

`pss4_project` is deliberately scoped to the local 1D periodic-stack FED
problem. It computes the local sidewall coefficient `h_W`; device-scale
quantities such as fin height `H`, aspect ratio, overlap length, and
footprint-normalized `h_eff` belong in the downstream geometry/thermal model.
Use `--meta KEY=VALUE` to record those external values in `run_metadata.json`
without coupling them into the optical stack solve.

## Features

- S-matrix formalism with Redheffer star product for arbitrary multi-layer stacking
- Analytical Bloch reflection coefficient via quadratic root selection
- Supports propagating and evanescent mode contributions
- Multiple material databases (SiO₂, Si₃N₄, SiC, Vacuum)
- Configurable frequency grid density for precision/performance trade-off

## Project Structure

```
pss4_project/
├── main.py              # Entry point / solver orchestrator
├── config_parser.py     # Parse stack configuration (.txt) files
├── materials.py         # Load material dielectric functions & frequency grids
├── grid_builder.py      # Build k_parallel discretization grid
├── scattering.py        # S-matrix core: Fresnel, Redheffer star product, Bloch reflection
├── fed_engine.py        # FED physics: transmission probability & spectral flux
├── post_processing.py   # Aggregate results, generate plots & reports
├── utils.py             # Physical constants (c, hbar, kB) & logging setup
├── materials/           # Material optical constant data files
│   ├── aligned_SiO2-Franta-300C.txt
│   ├── aligned_Si3N4-Luke(true).txt
│   ├── aligned_Vacuum.txt
│   ├── aligned_Si3N4-Philipp.txt
│   ├── aligned_SiC_Lorentz-sparse_53.txt
│   ├── aligned-SiO2_Franta-300C-sparse_53.txt
│   ├── aligned_Si3N4-Luke(true)-sparse_53.txt
│   └── aligned-Vacuum-sparse_53.txt
├── examples/            # Example input configuration files
│   ├── example_2layer.txt           # 2-layer: SiN + Vacuum
│   ├── example_4layer_coated.txt   # 4-layer: SiO2 coat + SiN + Vacuum + SiN
│   └── example_4layer_SiN_coat.txt # 4-layer: SiN coat + SiN + Vacuum + SiN
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

### From source

```bash
git clone <repo-url> pss4_project
cd pss4_project
pip install -r requirements.txt
```

### Dependencies

| Package   | Version  | Purpose                  |
|-----------|----------|--------------------------|
| numpy     | >= 1.20  | Numerical computation    |
| pandas    | >= 1.3   | Data handling & CSV I/O  |
| matplotlib| >= 3.4   | Plotting & visualization |

Requires Python >= 3.8.

## Quick Start

```bash
# Run with full frequency grid (2502 points)
python main.py examples/example_2layer.txt --temperature 300 --all-freq --out-dir results/

# Run with sparse frequency grid (53 points, faster)
python main.py examples/example_4layer_coated.txt --temperature 300 --out-dir results/
```

### Custom input and output directories

The stack configuration file is provided as the positional `input_path` argument,
and results can be written to any directory with `--out-dir`:

```bash
python main.py D:/project_inputs/coated_stack.txt \
  --temperature 300 \
  --out-dir D:/project_results/pss4/coated_stack
```

The output directory is created when needed. Use a separate output directory for
each independent run or parallel shard. The default output directory is
`results/`.

## Usage

### Command-line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_path` | Path to stack configuration file (required) | — |
| `--temperature` | Source temperature in Kelvin | None |
| `--all-freq` | Use full frequency grid (2502 points) | False |
| `--freq-skip N` | Use every `N`th frequency from the selected material table | `1` |
| `--freq-offset N` | Starting offset for `--freq-skip` subsets | `0` |
| `--freq-indices LIST_OR_FILE` | Explicit frequency indices as `0,5,10` or a text file | None |
| `--freq-values-file PATH` | Explicit omega values in rad/s; values must exist in the material tables | None |
| `--resume` | Skip omega rows already present in output CSVs and merge sorted unique outputs | False |
| `--out-dir` | Output directory | `results/` |
| `--meta KEY=VALUE` | Passive metadata tag written to `run_metadata.json`; may be repeated | None |

### Tau heatmaps

Generate a `tau_p(omega,k_parallel)` heatmap with the propagating/evanescent
boundary drawn using the shared light-line helper:

```bash
python scripts/visualize_tau_heatmaps.py examples/example_2layer.txt --polarization p --out-dir results/
```

Use `--no-prop-evan-boundary` to omit the curve. Boundary style can be adjusted
with `--boundary-color`, `--boundary-linestyle`, and `--boundary-linewidth`.

To post-process saved `tau_s` and `tau_p` grids into propagating and evanescent
transmission/spectral-flux contributions, see
[`docs/tau_prop_evan_postprocessing.md`](docs/tau_prop_evan_postprocessing.md).
The same guide documents the separate Bloch-mode classification for the exact
periodic input stack.

### Programmatic usage

```python
from main import run_solver

args = {
    'temperature': 300.0,
    'all_freq': True,
    'freq_skip': 2,       # Optional: skip factor for frequency grid
    'out_dir': 'results/',
    'metadata': {'W_um': '0.6667', 'H_um': '20.001', 'aspect_ratio': '30'}
}
run_solver('examples/example_2layer.txt', args)
```

### Frequency grid control and resumable refinement

The `freq_skip` parameter controls frequency grid density:

| freq_skip | Points | Use case |
|-----------|--------|----------|
| 1 | 2502 | Full resolution |
| 2 | 1251 | Standard |
| 6 | 417 | Quick scan |

For a rough-to-dense workflow, run a sparse subset first, then rerun the same
input into the same output directory with a denser subset and `--resume`.
Already-computed angular frequencies are skipped, and the output CSVs are
merged by `omega (rad/s)`:

```bash
# rough pass: every 12th dense-grid frequency
python main.py examples/example_2layer.txt --temperature 300 --all-freq --freq-skip 12 --out-dir results/case_a

# refinement pass: every 3rd dense-grid frequency, reusing the same output dir
python main.py examples/example_2layer.txt --temperature 300 --all-freq --freq-skip 3 --resume --out-dir results/case_a
```

Use `--freq-offset` to split one stride into independent parallel shards, for
example `--freq-skip 4 --freq-offset 0`, `1`, `2`, and `3`. Use distinct output
directories for parallel processes, or serialize `--resume` writes to the same
directory.

## Input File Format

Stack configuration files define the unit cell layer structure:

```
# layer_id,material,thickness,is_source,is_probe
PERIODIC_BOUNDARY
0,SiO2,0.050000,True,False
1,SiN,0.150000,True,False
2,Vacuum,0.100000,False,True
3,SiN,0.150000,True,False
PERIODIC_BOUNDARY
```

- `layer_id`: Integer layer identifier
- `material`: Material name (`SiO2`, `SiN`, `SiC`, `Vacuum`, `Silica`, `Si3N4`)
- `thickness`: Layer thickness in micrometers (μm)
- `is_source`: Whether this layer is a thermal source (`True`/`False`)
- `is_probe`: Whether this layer is the probe layer where flux is measured (`True`/`False`)

Optional group definitions (for per-group output):
```
# group: name=hot, ids=(0,4,8)
# group: name=cold, ids=(2,6)
```

## Supported Materials

| Keyword | File | Description |
|---------|------|-------------|
| `SiO2` / `Silica` | `aligned_SiO2-Franta-300C.txt` | SiO₂ (Franta model, 300°C) |
| `SiN` / `Si3N4` | `aligned_Si3N4-Luke(true).txt` | Si₃N₄ (Luke model) |
| `SiC` | `aligned_SiC_Lorentz-sparse_53.txt` | SiC (Lorentz model) |
| `Vacuum` | `aligned_Vacuum.txt` | Vacuum (ε = 1) |

## Output

For each probe layer, the solver produces:

- `P{id}.csv` — Raw per-source results (transmission, spectral flux vs frequency)
- `tot_P{id}.csv` — Aggregated results
- `tot_P{id}_plot.png` — Spectral flux plot (if temperature specified)
- `final_report.txt` — Summary with integrated heat transfer coefficient
- `run_metadata.json` - Solver grid/layer metadata plus passive external
  metadata supplied with `--meta`.

## Method

1. **Parse** input file → layer structure
2. **Load** material dielectric functions on shared frequency grid
3. **For each frequency** ω and **each k∥**:
   - Compute S-matrix for the unit cell (propagation + interfaces via Redheffer star product)
   - Compute Bloch reflection coefficients R_L, R_R (analytical quadratic solution)
   - Compute transmission probability τ(k∥, ω)
4. **Integrate** over k∥ → Φ(ω), then multiply by Planck derivative → spectral flux
5. **Integrate** over ω → heat transfer coefficient h_W

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
