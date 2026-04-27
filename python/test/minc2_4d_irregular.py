"""
Tests for irregular time-dimension support exposed through the Python
wrapper. Covers TDD cycles 5-7:

    5. fixture's time dim has .irregular True via representation_dims()
    6. fixture's time dim has .offsets equal to the known per-sample times
    7. defining + writing a new file with custom offsets round-trips
       offsets through the Python API
"""

import os
import tempfile
import unittest

import numpy as np

from minc2_simple import minc2_file, minc2_dim


# Matches /app/subproject/minc2-simple/test/test_4D_irregular_offsets.mnc
EXPECTED_OFFSETS = np.array(
    [0.0, 120.0, 300.0, 800.0, 1200.0, 1500.0], dtype=np.float64
)
FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "test", "test_4D_irregular_offsets.mnc",
)


def _find_time(dims):
    for i, d in enumerate(dims):
        if d.id == minc2_file.MINC2_DIM_TIME:
            return i
    raise AssertionError("no time dimension in {!r}".format(dims))


class TestIrregularRead(unittest.TestCase):
    """Cycles 5 and 6: read path through the Python wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.vol = minc2_file(FIXTURE)
        cls.dims = cls.vol.representation_dims()

    @classmethod
    def tearDownClass(cls):
        cls.vol.close()

    def test_irregular_flag(self):
        # Cycle 5
        t = _find_time(self.dims)
        self.assertTrue(
            self.dims[t].irregular,
            "fixture time dim should be irregular, got {!r}".format(self.dims[t]),
        )
        self.assertEqual(self.dims[t].length, len(EXPECTED_OFFSETS))

    def test_offsets_match(self):
        # Cycle 6
        t = _find_time(self.dims)
        offsets = self.dims[t].offsets
        self.assertIsNotNone(offsets, "irregular dim should expose offsets")
        np.testing.assert_allclose(offsets, EXPECTED_OFFSETS, rtol=0, atol=1e-9)

    def test_regular_dim_offsets_is_none(self):
        # Spatial dims must report offsets=None so callers can rely on truthiness.
        for d in self.dims:
            if d.id != minc2_file.MINC2_DIM_TIME:
                self.assertFalse(
                    d.irregular, "spatial dim should not be irregular"
                )
                self.assertIsNone(
                    d.offsets,
                    "regular dim should not carry offsets, got {!r}".format(d.offsets),
                )


class TestIrregularRoundTrip(unittest.TestCase):
    """Cycle 7: define+write+reopen round-trip via the Python wrapper."""

    def test_write_and_reread(self):
        custom = np.array(
            [0.5, 1.5, 4.0, 9.0, 16.0, 25.0], dtype=np.float64
        )
        # Build an irregular time dim plus three regular spatial dims.
        # We pass a numpy `offsets` array and rely on the wrapper to
        # marshal it into the C struct.
        dims = [
            dict(
                id=minc2_file.MINC2_DIM_TIME,
                length=len(custom),
                start=0.0,
                step=1.0,
                irregular=True,
                offsets=custom,
            ),
            dict(id=minc2_file.MINC2_DIM_Z, length=5, start=0.0, step=1.0),
            dict(id=minc2_file.MINC2_DIM_Y, length=5, start=0.0, step=1.0),
            dict(id=minc2_file.MINC2_DIM_X, length=5, start=0.0, step=1.0),
        ]

        path = tempfile.NamedTemporaryFile(
            prefix="minc2-irregular-", suffix=".mnc", delete=False
        ).name
        try:
            out = minc2_file()
            out.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
            out.create(path)
            data = np.arange(
                len(custom) * 5 * 5 * 5, dtype=np.float32
            ).reshape((len(custom), 5, 5, 5))
            out.save_complete_volume(data)
            out.close()

            back = minc2_file(path)
            try:
                back_dims = back.representation_dims()
                t = _find_time(back_dims)
                self.assertTrue(back_dims[t].irregular)
                np.testing.assert_allclose(
                    back_dims[t].offsets, custom, rtol=0, atol=1e-9
                )
            finally:
                back.close()
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestWidthsRoundTrip(unittest.TestCase):
    """Custom widths round-trip: define with explicit widths, write, reopen."""

    def test_custom_widths_roundtrip(self):
        custom_offsets = np.array(
            [0.5, 1.5, 4.0, 9.0, 16.0, 25.0], dtype=np.float64
        )
        custom_widths = np.array(
            [2.0, 3.0, 4.5, 1.0, 5.5, 0.5], dtype=np.float64
        )
        dims = [
            dict(
                id=minc2_file.MINC2_DIM_TIME,
                length=len(custom_offsets),
                start=0.0,
                step=1.0,
                irregular=True,
                offsets=custom_offsets,
                widths=custom_widths,
            ),
            dict(id=minc2_file.MINC2_DIM_Z, length=3, start=0.0, step=1.0),
            dict(id=minc2_file.MINC2_DIM_Y, length=3, start=0.0, step=1.0),
            dict(id=minc2_file.MINC2_DIM_X, length=3, start=0.0, step=1.0),
        ]

        path = tempfile.NamedTemporaryFile(
            prefix="minc2-widths-", suffix=".mnc", delete=False
        ).name
        try:
            out = minc2_file()
            out.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
            out.create(path)
            data = np.arange(
                len(custom_offsets) * 3 * 3 * 3, dtype=np.float32
            ).reshape((len(custom_offsets), 3, 3, 3))
            out.save_complete_volume(data)
            out.close()

            back = minc2_file(path)
            try:
                back_dims = back.representation_dims()
                t = _find_time(back_dims)
                self.assertTrue(back_dims[t].irregular)
                np.testing.assert_allclose(
                    back_dims[t].offsets, custom_offsets, rtol=0, atol=1e-9
                )
                np.testing.assert_allclose(
                    back_dims[t].widths, custom_widths, rtol=0, atol=1e-9
                )
            finally:
                back.close()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_minic2_dim_with_widths(self):
        """Define using minc2_dim namedtuple (not dict)."""
        custom_offsets = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        custom_widths = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        dims = [
            minc2_dim(
                id=minc2_file.MINC2_DIM_TIME,
                length=3,
                start=0.0,
                step=1.0,
                have_dir_cos=False,
                dir_cos=np.zeros(3, np.float64),
                irregular=True,
                offsets=custom_offsets,
                widths=custom_widths,
            ),
            minc2_dim(
                id=minc2_file.MINC2_DIM_Z, length=2, start=0.0, step=1.0,
                have_dir_cos=False, dir_cos=np.zeros(3, np.float64),
                irregular=False, offsets=None, widths=None,
            ),
            minc2_dim(
                id=minc2_file.MINC2_DIM_Y, length=2, start=0.0, step=1.0,
                have_dir_cos=False, dir_cos=np.zeros(3, np.float64),
                irregular=False, offsets=None, widths=None,
            ),
            minc2_dim(
                id=minc2_file.MINC2_DIM_X, length=2, start=0.0, step=1.0,
                have_dir_cos=False, dir_cos=np.zeros(3, np.float64),
                irregular=False, offsets=None, widths=None,
            ),
        ]

        path = tempfile.NamedTemporaryFile(
            prefix="minc2-widths-namedtuple-", suffix=".mnc", delete=False
        ).name
        try:
            out = minc2_file()
            out.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
            out.create(path)
            data = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape((3, 2, 2, 2))
            out.save_complete_volume(data)
            out.close()

            back = minc2_file(path)
            try:
                back_dims = back.representation_dims()
                t = _find_time(back_dims)
                self.assertTrue(back_dims[t].irregular)
                np.testing.assert_allclose(
                    back_dims[t].widths, custom_widths, rtol=0, atol=1e-9
                )
            finally:
                back.close()
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestFullCycle(unittest.TestCase):
    """Cycle 8: read fixture (dims + data), write a new file using those
    exact dims, reopen, assert both metadata and voxel data match."""

    def test_full_cycle(self):
        # 1. Read the source fixture: dims + voxel data.
        src = minc2_file(FIXTURE)
        try:
            src_dims = src.representation_dims()
            src_data = src.load_complete_volume(minc2_file.MINC2_FLOAT)
        finally:
            src.close()

        path = tempfile.NamedTemporaryFile(
            prefix="minc2-fullcycle-", suffix=".mnc", delete=False
        ).name
        try:
            # 2. Define + create + write a new file using the same dims.
            out = minc2_file()
            out.define(
                src_dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT
            )
            out.create(path)
            out.save_complete_volume(src_data)
            out.close()

            # 3. Reopen the new file and verify.
            back = minc2_file(path)
            try:
                back_dims = back.representation_dims()
                back_data = back.load_complete_volume(minc2_file.MINC2_FLOAT)
            finally:
                back.close()

            # Metadata: time dim still irregular with the same offsets.
            t_src = _find_time(src_dims)
            t_back = _find_time(back_dims)
            self.assertTrue(back_dims[t_back].irregular)
            self.assertEqual(back_dims[t_back].length, src_dims[t_src].length)
            np.testing.assert_allclose(
                back_dims[t_back].offsets,
                src_dims[t_src].offsets,
                rtol=0,
                atol=1e-9,
            )
            # Data: voxels survive the round-trip exactly (float32 → float32).
            np.testing.assert_array_equal(back_data, src_data)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
