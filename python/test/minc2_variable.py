import unittest
import numpy as N
import os
import tempfile

from minc2_simple import minc2_file, minc2_error


inputFile_4D = None


def setUpModule():
    global inputFile_4D
    # use the pre-existing 4D test file that has a time dimension variable
    inputFile_4D = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'test', 'test-nan-float_4D.mnc')
    if not os.path.exists(inputFile_4D):
        raise unittest.SkipTest("test-nan-float_4D.mnc not found")


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
        # create a new volume, write a variable, read it back
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="test-var-write-", suffix=".mnc", delete=False).name
        try:
            # create a simple 3D volume
            src = minc2_file(inputFile_4D)
            dims = src.store_dims()
            src.close()

            out = minc2_file()
            out.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
            out.create(tmpfile)
            out.setup_standard_order()
            # write zeros
            shape = [d.length for d in dims]
            data = N.zeros(shape, dtype=N.float32)
            out.save_complete_volume(data)

            # now read the time dimension variable we know exists after creating
            # from 4D dims, and overwrite it
            new_offsets = N.array([0.0, 2.0, 20.0], dtype=N.float64)
            out.write_variable("dimensions", "time", new_offsets)

            # read it back
            readback = out.read_variable("dimensions", "time")
            N.testing.assert_array_almost_equal(readback, new_offsets)

            out.close()
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


if __name__ == '__main__':
    unittest.main()

# kate: indent-mode python; indent-width 4; replace-tabs on;
