# Academic Description of the PSS4 Project

## 1. Purpose and Scope

`pss4_project` is a one-dimensional periodic fluctuational electrodynamics
(FED) solver for near-field radiative heat transfer across planar multilayer
unit cells. [REF needed: cite foundational fluctuational electrodynamics and
near-field radiative heat-transfer literature.] Its primary output is a local
sidewall radiative heat-transfer coefficient, denoted in the project
documentation as `h_W`. Device-scale quantities such as fin height `H`, width
`W`, aspect ratio `H/W`, overlap length, and footprint-normalized effective
heat-transfer coefficients are not part of the optical FED solve itself. They
are recorded as passive metadata or handled by downstream geometry and thermal
models in the generated batch folders.

The code should therefore be read as three coupled but separable layers:

1. a physical model construction layer, where geometry, assumptions, equations,
   and all physical quantities are defined;
2. a numerical algorithm layer, where those quantities are discretized and
   evaluated by mesh generation, S-matrix composition, Bloch-root selection,
   and quadrature;
3. a Python implementation layer, where stack files, material tables,
   resumable runs, output files, and batch metadata are managed.

## 2. A. Physical Model Construction and Quantity Definitions

### 2.1 Geometry and Model Domain

The simulated object is a planar unit cell bounded by two
`PERIODIC_BOUNDARY` markers in an input file. Each layer has a material,
thickness, source flag, and probe flag. A typical uncoated two-layer cell is

```text
PERIODIC_BOUNDARY
0,SiN,0.150000,True,False
1,Vacuum,0.100000,False,True
PERIODIC_BOUNDARY
```

The unit cell is repeated indefinitely along the stacking direction.
[REF needed: cite previous use of periodic multilayer or Bloch-boundary
descriptions in thermal radiation / photonic crystal heat transfer.] In-plane
translational symmetry is assumed, so every electromagnetic mode is labeled by
angular frequency, parallel wave vector, and polarization. The solver does not
model a laterally patterned grating with diffraction orders; it models a
stratified periodic medium.

The probe layer is required to be vacuum. The heat-transfer coefficient is
computed across that vacuum gap. Coatings are represented by inserting
additional material layers on either side of the vacuum layer. In batch studies,
the symbol `delta_um` denotes the original gap budget and `d_eff_um` denotes
the actual vacuum thickness after coating,

```text
d_eff_um = delta_um - 2 * coating_thickness_um.
```

This equation is a project convention for input construction, not a separate
electromagnetic approximation.

### 2.2 Quantity Definitions and Notation

All physical and numerical symbols used later in the algorithm section are
defined here so that the algorithm can refer back to the physical model without
redefining quantities.

| Symbol or name | Definition |
|---|---|
| `j` | layer index inside the periodic unit cell |
| `N` | number of layers in the unit cell |
| `p` | probe-layer index |
| `t_j` | thickness of layer `j` in meters after parsing |
| `d`, `d_gap` | vacuum probe-gap thickness used in the transmission formula |
| `delta_um` | original gap budget before coating, in micrometers |
| `d_eff_um` | actual vacuum gap after coatings, in micrometers |
| `coating_thickness_um` | coating thickness on each side of the vacuum gap |
| `H` | downstream device height metadata, not used in the optical solve |
| `W` | downstream device width metadata, not used in the optical solve |
| `H/W` | downstream aspect-ratio metadata |
| `omega` | angular frequency in rad/s |
| `omega_min` | smallest angular frequency included in a selected run |
| `T` | temperature in Kelvin for the linear-response Planck derivative |
| `c` | speed of light in vacuum |
| `hbar` | reduced Planck constant |
| `k_B` | Boltzmann constant |
| `epsilon_j(omega)` | complex relative permittivity of layer `j` |
| `epsilon_real`, `epsilon_imag` | real and imaginary parts of the material table entry |
| `k_parallel` | in-plane wave-vector magnitude |
| `kz_j` | normal wave-vector component in layer `j` |
| `kz_gap` | normal wave-vector component evaluated in the vacuum probe gap |
| `s`, `p` | transverse-electric and transverse-magnetic polarizations |
| `r_s`, `r_p` | Fresnel reflection coefficients for `s` and `p` polarization |
| `phase_j` | layer propagation factor `exp(i kz_j t_j)` |
| `S` | one-dimensional scattering block `(R11, T12, T21, R22)` |
| `A`, `B` | adjacent scattering blocks used in Redheffer composition |
| `R11`, `R22` | reflection amplitudes of a scattering block from left and right |
| `T12`, `T21` | transmission amplitudes of a scattering block from left to right and right to left |
| `R_L`, `R_R` | semi-infinite periodic reflection coefficients seen from the probe gap |
| `R` | generic unknown Bloch reflection coefficient in the quadratic solve |
| `D` | multiple-reflection denominator of the probe gap |
| `tau` | dimensionless mode transmission probability |
| `Phi(omega)` | polarization-summed transmission density after `k_parallel` integration |
| `Theta(omega,T)` | mean Planck oscillator energy |
| `dTheta/dT` | temperature derivative of `Theta` |
| `h_omega` | spectral heat-transfer coefficient density |
| `h_W` | final local sidewall heat-transfer coefficient |
| `k_min`, `k_max` | lower and upper limits of the discrete `k_parallel` mesh |
| `N_k` | number of `k_parallel` mesh points |
| `k_weights` | quadrature weights on the `k_parallel` mesh |

### 2.3 Material Model

Each material is described by a local, isotropic, complex dielectric function.
[REF needed: cite local dielectric-response approximation for FED and any
material-data source used for SiO2, Si3N4, and SiC.]

```text
epsilon_j(omega) = epsilon_real,j(omega) + i epsilon_imag,j(omega).
```

The material tables in `materials/` use three columns:

```text
omega(rad/s)    epsilon_real    epsilon_imag
```

Supported material aliases include:

| Input keyword | Material table family | Interpretation |
|---|---|---|
| `SiO2`, `Silica` | Franta SiO2 data | dielectric coating or layer |
| `SiN`, `Si3N4` | Luke Si3N4 data by default | silicon nitride layer |
| `SiC` | Lorentz SiC table | silicon carbide layer |
| `Vacuum` | epsilon approximately 1 | vacuum gap |

The material response is tabulated rather than fitted inside the solver. The
temperature supplied on the command line affects the Planck derivative used to
convert transmission into conductance, but the optical constants themselves are
loaded from fixed files. The present model therefore neglects temperature
dependence of dielectric functions unless new temperature-specific material
tables are supplied. [REF needed: cite any prior work or experimental database
supporting the selected temperature-independent material approximation.]

### 2.4 Electromagnetic Mode Construction

For layer `j`, the normal wave-vector component is

```text
kz_j = sqrt(epsilon_j(omega) * (omega/c)^2 - k_parallel^2).
```

The implemented branch condition is

```text
Im(kz_j) >= 0,
and if Im(kz_j) = 0 then Re(kz_j) >= 0.
```

This convention ensures decay away from interfaces for evanescent waves and
positive propagation direction for propagating waves. [REF needed: cite
standard branch-selection convention for stratified-medium scattering.]

At an interface between media `i` and `j`, the Fresnel reflection coefficients
are

```text
r_s = (kz_i - kz_j) / (kz_i + kz_j),

r_p = (epsilon_j * kz_i - epsilon_i * kz_j)
      / (epsilon_j * kz_i + epsilon_i * kz_j).
```

Both polarizations are included in the final transmission integral. [REF
needed: cite standard Fresnel-coefficient treatment for thermal radiation
through stratified media.]

The layer propagation factor is

```text
phase_j = exp(i kz_j t_j).
```

A scalar one-dimensional scattering block is represented as

```text
S = (R11, T12, T21, R22).
```

The scalar representation is valid here because the model keeps one channel for
each `(omega, k_parallel, polarization)` and does not include lateral
diffraction orders.

### 2.5 Periodic Environment and Transmission

The finite layer sequence inside the periodic boundaries is converted into an
effective left and right optical environment around the probe gap. These
environments are represented by Bloch reflection coefficients `R_L` and `R_R`
for semi-infinite periodic repetitions of the unit cell. [REF needed: cite
Bloch-mode or transfer/scattering-matrix treatment of semi-infinite periodic
multilayers.]

For the vacuum gap, the multiple-reflection denominator is

```text
D = 1 - R_L * R_R * exp(2 i kz_gap d).
```

The mode transmission probability follows the standard FED form. [REF needed:
cite Landauer/scattering formulation of near-field radiative heat transfer.]
For propagating modes,

```text
tau = (1 - |R_L|^2) * (1 - |R_R|^2) / |D|^2.
```

For evanescent modes,

```text
tau = 4 Im(R_L) Im(R_R) exp(-2 Im(kz_gap) d) / |D|^2.
```

The code clamps negative numerical roundoff in `tau` to zero.

### 2.6 Thermal Conductance Model

For each frequency, the polarization-summed transmission density is

```text
Phi(omega) = sum_pol integral_0^infinity
             tau_pol(omega, k_parallel) * k_parallel dk_parallel / (2 pi).
```

When a temperature `T` is supplied, the solver computes the linear-response
thermal conductance spectrum using the derivative of the Planck oscillator
energy. [REF needed: cite linear-response derivation for radiative thermal
conductance.]

```text
Theta(omega, T) = hbar * omega / (exp(hbar * omega / (k_B T)) - 1),

dTheta/dT = (hbar * omega)^2 exp(x)
            / (k_B T^2 (exp(x) - 1)^2),

x = hbar * omega / (k_B T).
```

The spectral heat-transfer coefficient is

```text
h_omega = (dTheta/dT) * Phi(omega) / (2 pi),
```

and the final local coefficient is

```text
h_W = integral h_omega d omega.
```

The reported integrated coefficient has units of `W/m^2/K`.

### 2.7 Mesh Quantities Defined by the Physical Length Scales

The discrete `k_parallel` mesh is an algorithmic object, but its limits are
defined from the physical gap length scale. The intended upper limit is

```text
k_max = max(1e10, 100 / d_gap),
```

and the lower limit used by the code is

```text
k_min = 1e-4 * omega_min / c.
```

The inverse-gap scaling is intended to retain high-`k_parallel` evanescent
contributions that dominate near-field transfer at small separations. [REF
needed: cite near-field evanescent-mode scaling and convergence practice for
large parallel wave vectors.]

### 2.8 Main Physical Approximations

The solver uses the following approximations. [REF needed: cite prior FED
modeling work using these assumptions, or justify deviations from more complete
models.]

- planar, laterally invariant layers;
- local, isotropic dielectric functions;
- coherent scattering through sharp interfaces;
- semi-infinite periodic continuation of the input unit cell;
- linear thermal response around the supplied temperature;
- no magnetic materials, nonlocality, surface roughness, finite lateral size,
  or explicit temperature-dependent material update;
- no direct coupling between the optical stack solve and downstream device
  parameters such as `H`, `W`, or `H/W`.

The `is_source` and `is_probe` flags primarily define which layers are counted
as emitting media and which vacuum layer is used for the measured gap. In the
current code path, grouped per-source outputs are an accounting convenience:
when groups are enabled, total transmission is distributed equally among source
layers rather than being computed from a layer-resolved absorption operator.

## 3. B. Algorithmic Structure

The algorithm section describes how the quantities defined in Section 2 are
evaluated. It intentionally avoids redefining the symbols; when a variable
appears below, its physical meaning is the one assigned in the physical-model
construction section.

### 3.1 Input Parsing and Frequency Selection

The numerical workflow begins by parsing a stack file into layer dictionaries.
Layer thickness values are converted to SI units during parsing. The material
loader then selects the angular-frequency grid from the tabulated material
files. The solver supports:

- sparse material grids for quick scans;
- full material grids with `--all-freq`;
- strided subsets with `--freq-skip` and `--freq-offset`;
- explicit frequency indices;
- explicit frequency values that must already exist in every material table.

No interpolation is performed. A requested angular frequency must be present in
all loaded material files.

### 3.2 Mesh Generation and Convergence Controls

The `k_parallel` mesh uses the `k_min`, `k_max`, `N_k`, and `k_weights`
quantities defined in Section 2.7. The current implementation uses a
logarithmic mesh with 1000 points and trapezoidal weights on that nonuniform
mesh.

Convergence in this project is controlled primarily by:

- number and placement of angular-frequency samples;
- `k_parallel` cutoff and mesh density;
- quality and density of tabulated dielectric data;
- stability of Bloch root selection for propagating and evanescent regimes.

Sparse grids are useful for scans, but final values should be checked against
denser frequency sampling and against changes in the `k_parallel` cutoff.
[REF needed: cite numerical convergence practices from prior near-field
radiative-transfer computations.]

### 3.3 Scattering Matrix Construction

For each selected `(omega, k_parallel, polarization)` tuple, the implementation
builds propagation and interface scattering blocks using the quantities defined
in Sections 2.4 and 2.5. Adjacent blocks are composed with the Redheffer star
product. [REF needed: cite Redheffer star product or scattering-matrix
stabilization literature for layered media.]

Using the adjacent scattering blocks `A` and `B` defined in Section 2.2, the
implemented composition is

```text
R11 = R11_A + T12_A R11_B (1 - R22_A R11_B)^-1 T21_A,

T12 = T12_A (1 - R11_B R22_A)^-1 T12_B,

T21 = T21_B (1 - R11_A R22_B)^-1 T21_A,

R22 = R22_B + T21_B R22_A (1 - R22_B R11_A)^-1 T12_B.
```

Because the physical model is scalar per polarization channel, these operations
are scalar complex operations rather than matrix inversions over multiple
diffraction channels.

### 3.4 Circular-Shift Periodic Reflection

Using the probe index `p` and cell size `N` defined in Section 2.2, the
algorithm forms two circularly shifted unit cells:

```text
right sequence: layers p+1 ... N-1, then 0 ... p
left sequence:  layers p ... N-1, then 0 ... p-1
```

The right sequence is used for `R_R`, and the left sequence is used for `R_L`.
Each shifted sequence is converted to a unit-cell S-matrix before applying the
Bloch reflection solve. [REF needed: cite or justify the circular-shift
construction for probe-centered periodic environments.]

### 3.5 Analytical Bloch Root Selection

The semi-infinite periodic reflection `R` is obtained from a quadratic
fixed-point condition rather than an iterative convergence loop. For the
forward reflection, the implemented relation is

```text
R = R11 + T12 R (1 - R22 R)^-1 T21,
```

which is rearranged as

```text
R22 R^2 + (T12 T21 - R11 R22 - 1) R + R11 = 0.
```

The backward reflection uses

```text
R11 R^2 + (T12 T21 - R11 R22 - 1) R + R22 = 0.
```

The code solves these equations analytically, not iteratively. The word
"convergence" in this project therefore mainly refers to grid convergence, not
fixed-point convergence of the Bloch reflection. Root selection uses the branch
of `kz_gap` already defined in the physical model:

- for evanescent waves, select the root with larger imaginary part of `R`;
- for propagating waves, select the root with smaller magnitude `|R|`;
- special degenerate cases with nearly zero quadratic coefficient are handled
  separately.

The physical justification for these root-selection criteria should be cited or
derived in a manuscript. [REF needed: cite Bloch reflection root-selection /
radiation-condition literature.]

### 3.6 Observable Calculation and Frequency Integration

For each frequency, the solver loops over both polarizations and all
`k_parallel` samples, computes `R_L`, `R_R`, evaluates `tau`, and integrates the
transmission density using `k_weights`. If temperature is supplied, the result
is converted to `h_omega` using `dTheta/dT`. After all frequencies are
processed, post-processing integrates `h_omega` over `omega` by a trapezoidal
rule to obtain `h_W`.

### 3.7 Resumable and Sharded Refinement

The command-line interface supports resumable frequency refinement. A sparse
run can be written first; later denser runs with `--resume` skip angular
frequencies already present in the output CSV and merge sorted unique rows.
`--freq-offset` allows a strided frequency set to be split into separate
parallel shards, provided those shards write to distinct output directories or
serialize writes to the same directory.

## 4. C. Coding and Implementation Layer

### 4.1 Main Modules

| File | Role |
|---|---|
| `main.py` | Command-line entry point and solver orchestration |
| `config_parser.py` | Stack-file parser and validation of probe/group definitions |
| `materials.py` | Material-file selection, frequency-grid selection, dielectric loading |
| `grid_builder.py` | Logarithmic `k_parallel` grid and quadrature weights |
| `scattering.py` | `kz`, Fresnel coefficients, propagation/interface S-matrices, Redheffer product, Bloch roots |
| `fed_engine.py` | Transmission formulas, Planck derivative, `k_parallel` observables |
| `post_processing.py` | Aggregation, plotting, final report generation |
| `utils.py` | Physical constants and logging setup |
| `examples/` | Minimal stack inputs for two-layer and coated four-layer cases |
| `materials/` | Dense and sparse dielectric-function tables |

The implementation relies only on `numpy`, `pandas`, and `matplotlib` beyond
the Python standard library.

### 4.2 Runtime Workflow

A typical run is

```bash
python main.py examples/example_2layer.txt --temperature 300 --all-freq --out-dir results/
```

The runtime sequence is:

1. parse CLI arguments;
2. parse stack geometry and optional source groups;
3. load material dielectric arrays on a shared frequency grid;
4. identify source and probe layers;
5. build the `k_parallel` grid;
6. for each probe, frequency, polarization, and `k_parallel`, compute the
   periodic reflections and transmission;
7. write raw probe CSVs;
8. aggregate totals and groups;
9. plot spectral flux if temperature is supplied;
10. write `final_report.txt` and `run_metadata.json`.

### 4.3 Input and Output Contract

Input files are plain text. Each non-comment layer row can be interpreted as

```text
layer_id, material, thickness_um, is_source, is_probe
```

The parser accepts comma-separated or whitespace-separated rows. A probe layer
must be vacuum. Optional group definitions are written after the periodic
boundary block, for example

```text
# group: name=hot, ids=(0,4,8)
```

For each probe `P{id}`, the solver may write:

| Output | Meaning |
|---|---|
| `P{id}.csv` | raw probe result versus angular frequency |
| `P{id}_S{source_id}.csv` | per-source accounting result when groups are enabled |
| `tot_P{id}.csv` | aggregated probe result |
| `sum_{group}_P{id}.csv` | group-summed result |
| `tot_P{id}_plot.png` | log-log spectral-flux plot |
| `final_report.txt` | parameter summary and integrated coefficient |
| `run_metadata.json` | solver metadata plus passive external geometry metadata |
| `execution.log` | detailed diagnostic log |

### 4.4 Batch Folders and Downstream Geometry

The generated batch directories contain input manifests, result summaries, and
plots for parameter sweeps. They use the PSS4 optical solver to obtain local
coefficients and then combine or compare them under external geometry models.
Examples include sweeps over

- `W_um`;
- aspect ratio `H/W`;
- `H_um = W_um * aspect_ratio`;
- coating thickness;
- effective gap `d_eff_um`;
- temperature;
- saturation rules for downstream `h_eff` curves.

These directories should not be confused with the core solver. They are
experiment-management and post-processing artifacts around the central local
FED calculation.

### 4.5 Implementation Caveats Worth Reviewing

The following points are important when using the code for publication-quality
results:

- The parser converts input thicknesses from micrometers to meters. In the
  current `run_solver` gap-detection path, the first vacuum-layer thickness is
  multiplied by `1e-6` again before being passed to `build_k_grid` and written
  to metadata. The transmission formula itself uses the already parsed probe
  thickness. This should be reviewed because it changes the `k_parallel`
  cutoff and metadata for the gap.
- Grouped per-source files currently partition the total transmission equally
  across source layers. They should not be interpreted as rigorous
  layer-resolved absorption or emission decomposition.
- The material loader requires exact frequency matches between tables. This is
  reproducible but places responsibility on the material-data preprocessing
  step.
- Sparse frequency grids are useful for scans, but final values of `h_W` should
  be checked against denser frequency sampling and `k_parallel` convergence.
- The model is scalar per polarization and assumes stratified media; it does
  not include lateral diffraction orders.

## 5. Summary

The project implements a coherent, planar, periodic-stack FED calculation. The
physical-model construction layer defines the geometry, all quantities,
assumptions, and heat-transfer equations. The algorithmic layer then discretizes
those quantities, composes scattering matrices, solves the Bloch reflection
quadratics, and performs quadrature. The coding layer provides a compact Python
pipeline for material loading, stack parsing, resumable sweeps, output
aggregation, plotting, and metadata generation.

This separation is essential: PSS4 computes the local optical sidewall
coefficient `h_W`; broader device-level quantities such as `H/W` saturation and
effective heat-transfer models are downstream analyses built around that local
coefficient.
