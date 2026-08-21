# TA examples

The examples in this directory serve different purposes. They are not a
single progression in which the newest file automatically replaces every
older validation script.

## Minimal / canonical example

- `ta_three_level_phase_cycling_v2_legacy_output_system_maker.py`
  is the current three-level systems-maker + TA recipe v2 bridge. Use its
  simulation, system, pulse, delay, and readout parameters as the baseline for
  focused TA comparisons.

## Regression / legacy validation

- `ta_three_level_intrinsic_response_phase_cycling_demo.py` preserves the
  original three-level phase-cycling workflow and legacy outputs.
- `ta_three_level_intrinsic_response_phase_cycling_demo_plus.py` adds report
  outputs around the original demo without replacing it.
- `ta_delay100fs_weak_field_analytic_ta_comparison.py` compares one saved
  numerical spectrum with a weak-field analytic reference.

## Extended / research examples

- `ta_three_level_phase_step_comparison.py` compares uniform N=2, N=3, N=4,
  N=8, and N=16 pump-phase grids at -100, 0, and +100 fs while keeping the
  canonical three-level parameters fixed.
- `ta_three_level_two_dimensional_phase_cycling.py` runs a full Cartesian
  pump/probe phase grid at one delay and projects only the phase-order channels
  declared by its JSON plan. The default plan evaluates S(0,0), S(0,1), and
  S(0,2) on N=8 grids, with an optional N=16 convergence grid. Its phase-case
  signal is the probe-heterodyne relative response
  `(A_pump_probe - A_probe_only) / A_probe_only`, where
  `A = omega * Im[P * conj(E_probe)]`.
- `ta_three_level_fixed_lo_detector_phase_cycling.py` is an analysis-only
  follow-up that separates the Hamiltonian interaction probe phase from a
  fixed phase-zero detector/LO probe. It compares exact detector intensity
  with the weak-signal heterodyne approximation, maps detector phase orders,
  and reports N=2 aliasing without rerunning saved propagations.
- `ta_four_level_readout_comparison_delay500fs.py` explores alternative
  four-level readout definitions at one long delay.
- `ta_harmonic_exciton_ladder_factorial_v3.py` explores EIS/PB/EID factorial
  cases. It is also the reference for split output, CSV/NPZ/JSON, and plotting
  organization, but not for the physical parameters of the three-level
  phase-step comparison.
- `plan_examples/*.json` contains plans consumed by the factorial workflows.
  It also contains the explicit two-dimensional pump/probe phase-cycling
  validation plan.

## Obsolete / unclear

- `ta_harmonic_exciton_ladder_factorial_v2.py` is retained as the predecessor
  of the split-output v3 workflow.
- `ta_delay100fs_lorentz_linear_comparison.py` is retained as the earlier,
  simplified Lorentz-line comparison.

No example in this inventory should be deleted or moved solely to simplify
the directory layout.
