from __future__ import annotations

import json
from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np

from qudpy_sjh.utils.serialization import json_safe, write_json


class JsonSerializationTests(unittest.TestCase):
    def test_numpy_complex_path_and_nonfinite_values(self) -> None:
        payload = json_safe(
            {
                "array": np.asarray([1.0 + 2.0j]),
                "path": Path("output") / "meta.json",
                "nan": np.float64(np.nan),
            }
        )

        self.assertEqual(payload["array"], [{"real": 1.0, "imag": 2.0}])
        self.assertEqual(payload["path"], str(Path("output") / "meta.json"))
        self.assertIsNone(payload["nan"])

    def test_nonfinite_real_ndarray_is_recursively_json_safe(self) -> None:
        payload = json_safe(np.asarray([1.0, np.nan, np.inf, -np.inf]))

        self.assertEqual(payload, [1.0, None, None, None])
        self.assertEqual(
            json.loads(json.dumps(payload, allow_nan=False)),
            [1.0, None, None, None],
        )

    def test_nonfinite_complex_ndarray_is_recursively_json_safe(self) -> None:
        payload = json_safe(np.asarray([1.0 + 2.0j, complex(np.inf, np.nan)]))

        self.assertEqual(
            payload,
            [
                {"real": 1.0, "imag": 2.0},
                {"real": None, "imag": None},
            ],
        )
        json.dumps(payload, allow_nan=False)

    def test_write_json_produces_standard_json(self) -> None:
        path = Path.cwd() / f".tmp_serialization_{uuid4().hex}.json"
        self.addCleanup(path.unlink, missing_ok=True)
        write_json(path, {"positive_infinity": float("inf")})

        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            {"positive_infinity": None},
        )


if __name__ == "__main__":
    unittest.main()
