from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from qutip import Qobj

from qudpy_sjh.experiments.transient_absorption import (
    ReadoutWindow,
    TAFullWindowCasePlan,
    TASettings,
    compute_absorption_readout,
    compute_pulse_centers,
    make_case_plan,
)
from qudpy_sjh.utils.core.results import DynamicsResult
from qudpy_sjh.utils.io import save_result_case


class _FakeDrive:
    name = "fake_lab_field"

    def __call__(self, times):
        return 0.01 * np.cos(np.asarray(times, dtype=float))

    def physical(self, times_fs):
        times = np.asarray(times_fs, dtype=float)
        return 0.01 * np.cos(0.05 * times)

    def to_dict(self):
        return {
            "class": "_FakeDrive",
            "name": self.name,
            "reference_field_MV_per_cm": 0.01,
            "metadata": {"E0_MV_per_cm": 0.01, "laser_energy_eV": 1.55},
        }

    def to_expr(self):
        return "0.01*cos(0.05*t_fs)"


def _small_settings() -> TASettings:
    return TASettings(
        t_start_fs=-120.0,
        t_end_fs=120.0,
        dt_fs=10.0,
        delays_fs=(0.0, 20.0),
        readout_start_fs=-40.0,
        readout_end_fs=40.0,
        pump_sigma_fs=8.0,
        probe_sigma_fs=8.0,
    )


def _fake_result(plan: TAFullWindowCasePlan | None = None) -> DynamicsResult:
    times_fs = np.linspace(-50.0, 50.0, 64)
    coherences = 0.02 * np.exp(1j * 0.12 * times_fs)
    states = []
    for coherence in coherences:
        rho = np.array(
            [
                [0.92, coherence],
                [np.conjugate(coherence), 0.08],
            ],
            dtype=np.complex128,
        )
        states.append(Qobj(rho))
    physical = None if plan is None else plan.physical_params
    return DynamicsResult(
        mode="lab_exact",
        times=times_fs,
        times_fs=times_fs,
        states=states,
        parameters=physical,
        physical_params=physical,
        metadata={"test": "full_window_ta_scaffold"},
        drive=_FakeDrive(),
        drive_dict=_FakeDrive().to_dict(),
        drive_expr=_FakeDrive().to_expr(),
        drive_name=_FakeDrive().name,
    )


class TransientAbsorptionScaffoldTests(unittest.TestCase):
    def test_probe_fixed_pulse_centers(self):
        self.assertEqual(compute_pulse_centers(delay_fs=200.0, probe_center_fs=0.0), (-200.0, 0.0))

    def test_settings_uses_probe_fixed_convention(self):
        settings = _small_settings()
        self.assertEqual(settings.pump_center_fs_for_delay(200.0), -200.0)

    def test_make_case_plan_is_full_window_only(self):
        settings = _small_settings()
        plan = make_case_plan(settings, 20.0)

        self.assertEqual(plan.delay_fs, 20.0)
        self.assertEqual(plan.pump_center_fs, -20.0)
        self.assertEqual(plan.probe_center_fs, 0.0)
        self.assertEqual(plan.simulation_window, (-120.0, 120.0))
        self.assertIsInstance(plan.readout_window, ReadoutWindow)
        plan.readout_window.validate_inside(plan.simulation_window)

        self.assertFalse(hasattr(plan, "pieces"))
        self.assertFalse(hasattr(plan, "active_windows"))
        self.assertFalse(hasattr(plan, "dark_backend"))

    def test_execute_returns_plain_dynamics_result(self):
        settings = _small_settings()
        plan = make_case_plan(settings, 0.0)
        calls = []

        def fake_run_case(physical_params, **kwargs):
            calls.append((physical_params, kwargs))
            return _fake_result(plan)

        result = plan.execute(run_case_fn=fake_run_case)

        self.assertIsInstance(result, DynamicsResult)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], plan.physical_params)
        self.assertNotEqual(type(result).__name__, "PieceDynamicsResultSeries")

    def test_absorption_readout_smoke(self):
        settings = _small_settings()
        plan = make_case_plan(settings, 0.0)
        result = _fake_result(plan)

        response = compute_absorption_readout(
            result,
            plan.readout_window,
            number_density_m3=settings.number_density_m3,
            window=settings.fft_window,
            subtract_mean=settings.subtract_mean,
            rel_threshold=settings.rel_threshold,
            zero_padding_factor=settings.zero_padding_factor,
        )

        self.assertIn("energy_eV", response)
        self.assertIn("neg_omega_im_P_over_E", response)
        self.assertGreater(response["energy_eV"].size, 0)
        self.assertEqual(response["readout_times_fs"].ndim, 1)

    def test_save_result_case_plain_result_outputs(self):
        settings = _small_settings()
        plan = make_case_plan(settings, 0.0, case_name="ta_plain_result")
        result = _fake_result(plan)

        with tempfile.TemporaryDirectory(
            dir=Path.cwd(),
            prefix="tmp_ta_scaffold_test_outputs_",
            ignore_cleanup_errors=True,
        ) as tmp:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(result.times_fs, result.populations()[:, 0])
            written = save_result_case(
                result,
                tmp,
                case_name="ta_plain_result",
                output_preview=True,
                preview_fig=fig,
                append_results_csv=False,
            )
            plt.close(fig)

            case_dir = Path(written["case_dir"])
            self.assertTrue((case_dir / "data" / "density.npz").exists())
            self.assertTrue((case_dir / "data" / "components.csv").exists())
            self.assertTrue((case_dir / "meta.json").exists())
            self.assertTrue((case_dir / "debug_meta.json").exists())
            self.assertTrue((case_dir / "figs" / "preview.png").exists())


if __name__ == "__main__":
    unittest.main()
