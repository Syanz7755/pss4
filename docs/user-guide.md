# PSS4 user guide

PSS4 reads a periodic-stack text input, evaluates the supported material tables,
and writes per-probe CSV data, optional spectral plots, a report, and run
metadata. The supported command-line options and input format are documented
in the project README; this file records the stable user-facing workflow.

```bash
python main.py examples/example_2layer.txt \
  --temperature 300 \
  --out-dir results/example_2layer
```

Use `--all-freq`, `--freq-skip`, `--freq-offset`, `--freq-indices`, or
`--freq-values-file` to control frequency selection. Use `--resume` to add
missing frequencies to an existing output directory. Use repeated `--meta
KEY=VALUE` options only for passive run labels; they do not change the solver.

The reusable solver is at the project root. Scripts under `scripts/` are
workflow helpers for sweeps, plots, or monitoring and should be treated as
optional tools. Large run directories and validation archives are data, not
required source files; keep them outside a fresh checkout when possible.

The repository also contains longer technical/manuscript notes. They are
reference material rather than required operating instructions; users only
need the README, this guide, and the relevant script `--help` output.
