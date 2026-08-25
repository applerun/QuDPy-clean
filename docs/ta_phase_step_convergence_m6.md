# TA phase-step convergence (Milestone 6)

## Scope

This validation uses the canonical workflow:

```text
System -> TAPrePCRecipe -> SimRes -> ReadoutPlan -> deltaT/T
       -> project_phase_orders -> save_projected_result
```

It does not use the historical heavy runner or TA-specific projected wrappers.

## Setup

- Three levels: energies `(0, 1.55, 3.25) eV`; adjacent dipoles `5 D`, `9 D`.
- Pure dephasing: level 1 `Tphi=120 fs`; level 2 `Tphi=100 fs`.
- Pump: Gaussian, `0.30 MV/cm`, `1.55 eV`, `sigma=12 fs`.
- Probe: Gaussian, `0.008 MV/cm`, `1.62 eV`, `sigma=7 fs`.
- Propagation: `lab_exact`, `[-1500, 1500] fs`, `dt=0.2 fs`.
- Delays: `T=(-100, 0, +100) fs`, with `pump_center=probe_center-T`.
- Phase dimensions: independent pump and probe interaction phases. The fixed
  phase-zero probe LO is excluded from the PhaseGrid.
- Grids: uniform `N x N`, for `N=2,3,4`.
- Physical target: `m={"pump": 0, "probe": 1}` for every N.
- Projection: pathway `exp(-i m dot phi)`, weight `exp(+i m dot phi)`.
- Readout: full fixed-LO detector, `emitted_field_scale=1`, followed by
  `deltaT/T=(I_on-I_off)/I_off`; relative denominator threshold `1e-8`.
- Comparison window: `1.4-1.8 eV`; N4 is only the highest-N numerical
  reference, not ground truth.

## Results

All 1267 bins in the comparison window were jointly valid for every pair.

| T (fs) | max abs S2 | max abs S3 | max abs S4 | max error N2-N4 | max error N3-N4 | relative L2 N2-N4 | relative L2 N3-N4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -100 | 4.205e-6 | 1.657e-6 | 2.438e-6 | 3.506e-6 | 1.067e-6 | 1.6966 | 0.4102 |
| 0 | 1.369e-5 | 4.807e-6 | 4.711e-6 | 9.385e-6 | 7.904e-7 | 1.4278 | 0.0906 |
| +100 | 1.124e-5 | 5.554e-6 | 5.984e-6 | 6.059e-6 | 1.162e-6 | 0.9872 | 0.0835 |

N2 has negligible projected imaginary amplitude (below `1e-20`) while N3/N4
have imaginary amplitudes of order `1e-6`. This is not numerical convergence:
on a two-step grid, physical probe orders `+1` and `-1` are the same modulo-2
class. N4 gives equal maximum amplitudes for `S_0_1` and its signed-equivalent
`S_0_3` class (`5.984e-6` over all delays/window), making the alias mechanism
explicit. `S_0_2` reaches `6.203e-7`; `S_0_0` is larger but is a separate order,
not assumed to be a contaminant.

Under this tested regime, N2 is not sufficient for the physical S(0,1)
channel. N3 agrees much more closely with N4 at 0 and +100 fs but retains a
0.410 relative-L2 difference at -100 fs. Therefore this run does not establish
N3 as generally converged. Higher-order content, field strength, pulse shape,
system, detector convention, and delay can change the required phase count.

## Detector check

The saved N4 SimRes cases were read once with full and weak `ReadoutPlan`
objects. No solver rerun was required. Full versus weak S(0,1) has relative-L2
`0.001666`, with maxima `5.984e-6` and `5.972e-6`, respectively. Both modes use
the same Recipe/postprocess axes; the phase-step conclusion is not driven by
the full detector quadratic term.

## Artifacts

Reproducible command:

```powershell
conda --no-plugins run -n quantum python bin/examples/ta/ta_three_level_canonical_phase_step_convergence.py
```

Generated outputs are intentionally untracked under:

```text
bin/optical_bloch_plots/ta_three_level_canonical_phase_step_convergence/
```

The directory contains per-case SimRes checkpoints, per-N projected NPZ/JSON,
`summary.csv`, `metadata.json`, `spectra.npz`, `N4_fourier_content.csv`, and
three delay overlays.

## M7 architecture status

The historical heavy runner, projected wrappers, TA v1/v2 plans, and embedded
single-run readout path were removed from the active tree in M7. Git history is
the compatibility archive. The validated M6 workflow remains:

```text
TAPrePCRecipe -> TAPrePCObservable -> project_phase_orders
```

`TAPrePCObservable` remains the thin recipe boundary carrying named axes,
denominator masks, and diagnostics.
