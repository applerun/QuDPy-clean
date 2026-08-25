# TA examples

This directory contains only active examples. Historical phase-cycling and TA
v1/v2 scripts were removed in M7 and remain available from Git history.

## Canonical workflow

- `ta_three_level_canonical_phase_step_convergence.py` is the concise reference
  for `System -> TAPrePCRecipe -> SingleRunPlan -> ReadoutPlan -> S(phi) ->
  project_phase_orders -> save/load_projected_result`. It runs the validated
  fixed-LO N=2/3/4 comparison at T=-100, 0, and +100 fs.
- `ta_harmonic_exciton_ladder_factorial_v3.py` applies the same recipe-first
  workflow to an outer EIS/PB/EID System scan. The scan axes stay outside the
  TA recipe. Its JSON plans control System cases, not phase-cycling runners.

## Independent research examples

- `ta_four_level_readout_comparison_delay500fs.py` compares several readout
  definitions for a distinct four-level physical model.

The canonical Fourier convention is
`S_m = mean[S(phi) * exp(+i m dot phi)]`; target vectors denote physical integer
phase orders. Pulse phases enter the Hamiltonian through physical fields. A
fixed external detector/LO is not a phase-grid axis.

Generated scientific outputs belong under `bin/optical_bloch_plots/` and are
not source examples.
