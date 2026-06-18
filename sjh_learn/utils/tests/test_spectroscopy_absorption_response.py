from __future__ import annotations

import unittest

import numpy as np

from sjh_learn.utils.spectroscopy import diagnose_uniform_time_axis, lab_frame_absorption_response


class SpectroscopyAbsorptionResponseTests(unittest.TestCase):
    def test_diagnose_uniform_time_axis(self):
        time_fs = np.linspace(-10.0, 10.0, 21)
        diagnostics = diagnose_uniform_time_axis(time_fs)

        self.assertEqual(diagnostics["n"], 21)
        self.assertEqual(diagnostics["t0_fs"], -10.0)
        self.assertEqual(diagnostics["t1_fs"], 10.0)
        self.assertTrue(diagnostics["is_uniform"])
        self.assertAlmostEqual(diagnostics["median_dt_fs"], 1.0)

    def test_diagnose_nonuniform_time_axis(self):
        diagnostics = diagnose_uniform_time_axis(np.array([0.0, 1.0, 2.2, 3.0]))

        self.assertFalse(diagnostics["is_uniform"])
        self.assertEqual(diagnostics["n"], 4)
        self.assertAlmostEqual(diagnostics["min_dt_fs"], 0.8)
        self.assertAlmostEqual(diagnostics["max_dt_fs"], 1.2)

    def test_lab_frame_absorption_response_smoke(self):
        time_fs = np.linspace(0.0, 63.0, 64)
        omega = 0.4
        field = np.cos(omega * time_fs)
        polarization = (0.2 + 0.05j) * field + 0.01j * np.sin(omega * time_fs)

        response = lab_frame_absorption_response(
            time_fs=time_fs,
            polarization_C_per_m2=polarization,
            field=field,
            window="hann",
            subtract_mean=True,
            return_intermediates=True,
        )

        self.assertIn("energy_eV", response)
        self.assertIn("omega_fs_inv", response)
        self.assertIn("absorption", response)
        self.assertIn("omega_im_P_over_E", response)
        self.assertIn("P_omega", response)
        self.assertIn("E_omega", response)
        self.assertIn("metadata", response)
        np.testing.assert_allclose(response["absorption"], response["omega_im_P_over_E"])
        self.assertTrue(response["metadata"]["time_axis"]["is_uniform"])
        self.assertGreater(response["energy_eV"].size, 0)

    def test_lab_frame_absorption_response_rejects_nonuniform_time_axis(self):
        time_fs = np.array([0.0, 1.0, 2.0, 4.0])
        signal = np.ones_like(time_fs)

        with self.assertRaisesRegex(ValueError, "uniformly sampled time axis"):
            lab_frame_absorption_response(
                time_fs=time_fs,
                polarization_C_per_m2=signal,
                field=signal,
            )

    def test_no_piecewise_dark_imports(self):
        import sjh_learn.utils.spectroscopy.absorption_spectra as spectra

        names = set(vars(spectra))
        self.assertNotIn("PieceDynamicsResultSeries", names)
        self.assertNotIn("piecewise", names)
        self.assertNotIn("dark_propagation", names)


if __name__ == "__main__":
    unittest.main()
