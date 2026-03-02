import unittest
import numpy as N
import os
import shutil
import tempfile

from minc2_simple import minc2_file, minc2_error


inputFile_4D = None
inputFile_3D = None


def setUpModule():
    global inputFile_4D, inputFile_3D
    # use the pre-existing 4D test file that has a time dimension variable
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'test')
    inputFile_4D = os.path.join(test_dir, 'test-nan-float_4D.mnc')
    if not os.path.exists(inputFile_4D):
        raise unittest.SkipTest("test-nan-float_4D.mnc not found")
    inputFile_3D = os.path.join(test_dir, 'test-nan-float-history.mnc')
    if not os.path.exists(inputFile_3D):
        raise unittest.SkipTest("test-nan-float-history.mnc not found")


class TestVariableRead(unittest.TestCase):
    """Test the variable read API via Python bindings."""

    def test_variable_ndims(self):
        """Test querying number of dimensions for a variable."""
        f = minc2_file(inputFile_4D)
        ndims = f.variable_ndims("dimensions", "time")
        self.assertEqual(ndims, 1)
        f.close()

    def test_variable_dims(self):
        """Test querying dimension sizes for a variable."""
        f = minc2_file(inputFile_4D)
        dims = f.variable_dims("dimensions", "time")
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims[0], 3)
        f.close()

    def test_variable_type(self):
        """Test querying data type of a variable."""
        f = minc2_file(inputFile_4D)
        vtype = f.variable_type("dimensions", "time")
        self.assertEqual(vtype, minc2_file.MINC2_DOUBLE)
        f.close()

    def test_read_variable_full(self):
        """Test reading a complete variable."""
        f = minc2_file(inputFile_4D)
        data = f.read_variable("dimensions", "time")
        self.assertEqual(data.shape, (3,))
        self.assertEqual(data.dtype, N.float64)
        N.testing.assert_array_almost_equal(data, [0.0, 1.0, 10.0])
        f.close()

    def test_read_variable_hyperslab(self):
        """Test reading a hyperslab from a variable."""
        f = minc2_file(inputFile_4D)
        # read only the last 2 elements
        data = f.read_variable("dimensions", "time", start=[1], count=[2])
        self.assertEqual(data.shape, (2,))
        N.testing.assert_array_almost_equal(data, [1.0, 10.0])
        f.close()

    def test_read_variable_type_conversion(self):
        """Test reading a variable with type conversion."""
        f = minc2_file(inputFile_4D)
        # read as float32 instead of native float64
        data = f.read_variable("dimensions", "time",
                               data_type=minc2_file.MINC2_FLOAT)
        self.assertEqual(data.dtype, N.float32)
        N.testing.assert_array_almost_equal(data, [0.0, 1.0, 10.0], decimal=5)
        f.close()

    def test_variable_nonexistent(self):
        """Test that accessing a nonexistent variable raises an error."""
        f = minc2_file(inputFile_4D)
        with self.assertRaises(minc2_error):
            f.variable_ndims("dimensions", "nonexistent")
        f.close()


class TestVariableWrite(unittest.TestCase):
    """Test the variable write API via Python bindings."""

    def test_write_variable(self):
        """Test writing to a variable and reading back."""
        # copy the input file and modify the time dimension offsets
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="test-var-write-", suffix=".mnc", delete=False).name
        try:
            shutil.copy2(inputFile_4D, tmpfile)

            f = minc2_file()
            f.open_rdwr(tmpfile)

            # read original time offsets
            orig = f.read_variable("dimensions", "time")
            N.testing.assert_array_almost_equal(orig, [0.0, 1.0, 10.0])

            # overwrite with new values
            new_offsets = N.array([0.0, 2.0, 20.0], dtype=N.float64)
            f.write_variable("dimensions", "time", new_offsets)

            # read back and verify
            readback = f.read_variable("dimensions", "time")
            N.testing.assert_array_almost_equal(readback, new_offsets)

            f.close()

            # reopen and verify persistence
            f2 = minc2_file(tmpfile)
            readback2 = f2.read_variable("dimensions", "time")
            N.testing.assert_array_almost_equal(readback2, new_offsets)
            f2.close()
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


class TestOpenRdwr(unittest.TestCase):
    """Test the open_rdwr mode for read-write file access."""

    def test_open_rdwr_reads_volume(self):
        """Verify open_rdwr can read voxel data just like open."""
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="test-rdwr-read-", suffix=".mnc", delete=False).name
        try:
            shutil.copy2(inputFile_4D, tmpfile)

            # read via normal open
            f_ro = minc2_file(tmpfile)
            f_ro.setup_standard_order()
            dims_ro = f_ro.representation_dims()
            data_ro = f_ro.load_complete_volume(minc2_file.MINC2_FLOAT)
            f_ro.close()

            # read via open_rdwr
            f_rw = minc2_file()
            f_rw.open_rdwr(tmpfile)
            f_rw.setup_standard_order()
            dims_rw = f_rw.representation_dims()
            data_rw = f_rw.load_complete_volume(minc2_file.MINC2_FLOAT)
            f_rw.close()

            # both should return identical results
            self.assertEqual(len(dims_ro), len(dims_rw))
            for d_ro, d_rw in zip(dims_ro, dims_rw):
                self.assertEqual(d_ro.id, d_rw.id)
                self.assertEqual(d_ro.length, d_rw.length)
            N.testing.assert_array_equal(data_ro, data_rw)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    def test_open_rdwr_writes_volume(self):
        """Verify open_rdwr allows writing voxel data."""
        from minc2_simple import minc2_dim
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="test-rdwr-write-", suffix=".mnc", delete=False).name
        try:
            # create a simple float volume (no slice scaling)
            dims = [
                minc2_dim(id=minc2_file.MINC2_DIM_Z, length=4, start=0.0,
                          step=1.0, have_dir_cos=0, dir_cos=N.zeros(3)),
                minc2_dim(id=minc2_file.MINC2_DIM_Y, length=4, start=0.0,
                          step=1.0, have_dir_cos=0, dir_cos=N.zeros(3)),
                minc2_dim(id=minc2_file.MINC2_DIM_X, length=4, start=0.0,
                          step=1.0, have_dir_cos=0, dir_cos=N.zeros(3)),
            ]
            f = minc2_file()
            f.define(dims,
                     store_type=minc2_file.MINC2_FLOAT,
                     representation_type=minc2_file.MINC2_FLOAT)
            f.create(tmpfile)
            f.setup_standard_order()
            initial = N.ones((4, 4, 4), dtype=N.float32)
            f.save_complete_volume(initial)
            f.close()

            # reopen read-write, overwrite volume with zeros
            f = minc2_file()
            f.open_rdwr(tmpfile)
            f.setup_standard_order()

            data = f.load_complete_volume(minc2_file.MINC2_FLOAT)
            N.testing.assert_array_almost_equal(data, initial)

            zeroed = N.zeros((4, 4, 4), dtype=N.float32)
            f.save_complete_volume(zeroed)
            f.close()

            # reopen read-only and confirm zeros
            f2 = minc2_file(tmpfile)
            f2.setup_standard_order()
            readback = f2.load_complete_volume(minc2_file.MINC2_FLOAT)
            f2.close()

            N.testing.assert_array_almost_equal(readback, zeroed)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    def test_open_readonly_rejects_variable_write(self):
        """Verify that writing a variable via read-only open raises an error."""
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="test-rdwr-reject-", suffix=".mnc", delete=False).name
        try:
            shutil.copy2(inputFile_4D, tmpfile)

            f = minc2_file(tmpfile)  # read-only
            new_offsets = N.array([0.0, 2.0, 20.0], dtype=N.float64)
            with self.assertRaises(minc2_error):
                f.write_variable("dimensions", "time", new_offsets)
            f.close()
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


if __name__ == '__main__':
    unittest.main()

# kate: indent-mode python; indent-width 4; replace-tabs on;
