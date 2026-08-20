from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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

    def test_write_json_produces_standard_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            write_json(path, {"positive_infinity": float("inf")})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"positive_infinity": None},
            )


if __name__ == "__main__":
    unittest.main()
