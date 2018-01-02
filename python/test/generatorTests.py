# based on https://github.com/Mouse-Imaging-Centre/pyminc/blob/master/test/generatorTests.py

import unittest
import numpy as N
import os
import subprocess
import tempfile

from minc2_simple import minc2_file,minc2_xfm,minc2_error


def setUpModule():
    global outputFilename,emptyFilename,newFilename,inputFile_byte,inputFile_short,inputFile_int
    global inputFile_float,inputFile_double,inputFile_ubyte,inputFile_ushort,inputFile_uint
    global inputVector,input3DdirectionCosines,outputXfmFilename1,outputXfmFilename2,outputXfmFilename3
    
    
    outputFilename = tempfile.NamedTemporaryFile(prefix="test-out-", suffix=".mnc").name
    emptyFilename = tempfile.NamedTemporaryFile(prefix="test-empty-", suffix=".mnc").name
    newFilename = tempfile.NamedTemporaryFile(prefix="test-new-volume-", suffix=".mnc").name

    inputFile_byte = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_byte, '-osigned', '-obyte', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_short = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_short, '-osigned', '-oshort', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_int = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_int, '-oint', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_float = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_float, '-ofloat', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_double = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_double, '-odouble', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_ubyte = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_ubyte, '-ounsigned', '-obyte', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_ushort = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_ushort, '-ounsigned', '-oshort', '-input', '/dev/urandom', '100', '150', '125'])

    inputFile_uint = tempfile.NamedTemporaryFile(prefix="test-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputFile_uint, '-ounsigned', '-oint', '-input', '/dev/urandom', '100', '150', '125'])


    inputVector = tempfile.NamedTemporaryFile(prefix="test-vector-", suffix=".mnc").name
    subprocess.check_call(['rawtominc', inputVector, '-input', '/dev/urandom', '3', '100', '150', '125',
                        '-dimorder', 'vector_dimension,xspace,yspace,zspace'])

    input3DdirectionCosines = tempfile.NamedTemporaryFile(prefix="test-3d-direction-cosines", suffix=".mnc").name
    subprocess.check_call(['rawtominc', input3DdirectionCosines, '-input', '/dev/urandom', '100', '150', '125',
                        '-xdircos',  '0.9305326623',   '0.1308213523', '0.34202943789', 
                        '-ydircos', '-0.1958356912',  '0.96692346178', '0.16316734231',
                        '-zdircos', '-9.3093890238', '-0.21882376893', '0.92542348732'])

    # testing for applying transformations to coordinates:
    outputXfmFilename1 = tempfile.NamedTemporaryFile(prefix="test-xfm-1", suffix=".xfm").name
    outputXfmFilename2 = tempfile.NamedTemporaryFile(prefix="test-xfm-2", suffix=".xfm").name
    outputXfmFilename3 = tempfile.NamedTemporaryFile(prefix="test-xfm-3", suffix=".xfm").name

    subprocess.check_call(["param2xfm", "-center", '2.21', '-3.765', '4.09', "-translation", '1.23', '6.4', '-7.8', "-scales", '0.2', '4.3', '-3', outputXfmFilename1])
    subprocess.check_call(["param2xfm", "-center", '-23.98', '0.46', '9.5', "-translation", '0.0', '-46', '89.3', "-scales", '10', '7.33', '84', outputXfmFilename2])
    subprocess.check_call(["xfmconcat", outputXfmFilename1, outputXfmFilename2, outputXfmFilename3])



def tearDownModule():
    os.remove(inputFile_byte)
    os.remove(inputFile_short)
    os.remove(inputFile_int)
    os.remove(inputFile_float)
    os.remove(inputFile_double)
    os.remove(inputFile_ubyte)
    os.remove(inputFile_ushort)
    os.remove(inputFile_uint)
    os.remove(inputVector)
    os.remove(input3DdirectionCosines)
    
    if os.path.exists(outputFilename):
        os.remove(outputFilename)
    if os.path.exists(newFilename):
        os.remove(newFilename)
    
    os.remove(outputXfmFilename1)
    os.remove(outputXfmFilename2)
    os.remove(outputXfmFilename3)

class TestFromFile(unittest.TestCase):
    """test the minc2_file reading using numpy"""
    def testFromFileError(self):
        """attempting to load a garbage file should raise exception"""
        self.assertRaises(minc2_error, minc2_file, "garbage.mnc")
    def testFromFileDataTypeByte(self):
        """ensure byte data is read as float by default"""
        v = minc2_file(inputFile_byte)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeShort(self):
        """ensure short data is read as float by default"""
        v = minc2_file(inputFile_short)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeInt(self):
        """ensure int data is read as float by default"""
        v = minc2_file(inputFile_int)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeFloat(self):
        """ensure float data is read as float by default"""
        v = minc2_file(inputFile_float)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeDouble(self):
        """ensure double data is read as float"""
        v = minc2_file(inputFile_double)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float64')
    def testFromFileDataTypeUByte(self):
        """ensure unsigned byte data is read as float by default"""
        v = minc2_file(inputFile_ubyte)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeUShort(self):
        """ensure unsigned short data is read as float by default"""
        v = minc2_file(inputFile_ushort)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataTypeUInt(self):
        """ensure unsigned int data is read as float by default"""
        v = minc2_file(inputFile_uint)
        dt = v.representation_dtype()
        v.close()
        self.assertEqual(dt, 'float32')
    def testFromFileDataByte(self):
        """ensure that byte data is read correct with a precision of 8 decimals on a call to aveage()"""
        v = minc2_file(inputFile_byte)
        a = N.average(v.load_complete_volume('float64'))
        v.close()
        pipe = os.popen("mincstats -mean -quiet %s" % inputFile_byte, "r")
        output = float(pipe.read())
        pipe.close()
        self.assertAlmostEqual(a, output, 8)
    def testFromFileDataShort(self):
        """ensure that short data is read correct with a precision of 8 decimals on a call to aveage()"""
        v = minc2_file(inputFile_short)
        a = N.average(v.load_complete_volume('float64'))
        v.close()
        pipe = os.popen("mincstats -mean -quiet %s" % inputFile_short, "r")
        output = float(pipe.read())
        pipe.close()
        self.assertAlmostEqual(a, output, 8)
    def testFromFileDataInt(self):
        """ensure that int data is read correct with a precision of 8 decimals on a call to aveage()"""
        v = minc2_file(inputFile_int)
        a = N.average(v.load_complete_volume('float64'))
        v.close()
        pipe = os.popen("mincstats -mean -quiet %s" % inputFile_int, "r")
        output = float(pipe.read())
        pipe.close()
        self.assertAlmostEqual(a, output, 8)
    def testFromFileDataFloat(self):
        """ensure that float data is read correct with a precision of 8 decimals on a call to aveage()"""
        v = minc2_file(inputFile_float)
        a = N.average(v.load_complete_volume('float64'))
        v.close()
        pipe = os.popen("mincstats -mean -quiet %s" % inputFile_float, "r")
        output = float(pipe.read())
        pipe.close()
        self.assertAlmostEqual(a, output, 8)
    def testFromFileDataDouble(self):
        """ensure that double data is read correct with a precision of 8 decimals on a call to aveage()"""
        v = minc2_file(inputFile_double) 
        a = N.average(v.data)
        v.close()
        pipe = os.popen("mincstats -mean -quiet %s" % inputFile_double, "r")
        output = float(pipe.read())
        pipe.close()
        self.assertAlmostEqual(a, output, 8)

try:
    import torch # this is going to work only if torch is present
    
    class TestFromFileTensor(unittest.TestCase):
        """test the minc2_file reading using pytorch tensor"""
        def testFromFileDataTypeByte(self):
            """ensure byte data is read as float by default"""
            v = minc2_file(inputFile_byte)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeShort(self):
            """ensure short data is read as float by default"""
            v = minc2_file(inputFile_short)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeInt(self):
            """ensure int data is read as float by default"""
            v = minc2_file(inputFile_int)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeFloat(self):
            """ensure float data is read as float by default"""
            v = minc2_file(inputFile_float)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeDouble(self):
            """ensure double data is read as float"""
            v = minc2_file(inputFile_double)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.DoubleTensor')
        def testFromFileDataTypeUByte(self):
            """ensure unsigned byte data is read as float by default"""
            v = minc2_file(inputFile_ubyte)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeUShort(self):
            """ensure unsigned short data is read as float by default"""
            v = minc2_file(inputFile_ushort)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataTypeUInt(self):
            """ensure unsigned int data is read as float by default"""
            v = minc2_file(inputFile_uint)
            dt = v.representation_dtype_tensor()
            v.close()
            self.assertEqual(dt, 'torch.FloatTensor')
        def testFromFileDataByte(self):
            """ensure that byte data is read correct with a precision of 8 decimals on a call to aveage()"""
            v = minc2_file(inputFile_byte)
            a = v.load_complete_volume_tensor('torch.DoubleTensor').mean()
            v.close()
            pipe = os.popen("mincstats -mean -quiet %s" % inputFile_byte, "r")
            output = float(pipe.read())
            pipe.close()
            self.assertAlmostEqual(a, output, 8)
        def testFromFileDataShort(self):
            """ensure that short data is read correct with a precision of 8 decimals on a call to aveage()"""
            v = minc2_file(inputFile_short)
            a = v.load_complete_volume_tensor('torch.DoubleTensor').mean()
            v.close()
            pipe = os.popen("mincstats -mean -quiet %s" % inputFile_short, "r")
            output = float(pipe.read())
            pipe.close()
            self.assertAlmostEqual(a, output, 8)
        def testFromFileDataInt(self):
            """ensure that int data is read correct with a precision of 8 decimals on a call to aveage()"""
            v = minc2_file(inputFile_int)
            a = v.load_complete_volume_tensor('torch.DoubleTensor').mean()
            v.close()
            pipe = os.popen("mincstats -mean -quiet %s" % inputFile_int, "r")
            output = float(pipe.read())
            pipe.close()
            self.assertAlmostEqual(a, output, 8)
        def testFromFileDataFloat(self):
            """ensure that float data is read correct with a precision of 8 decimals on a call to aveage()"""
            v = minc2_file(inputFile_float)
            a = v.load_complete_volume_tensor('torch.DoubleTensor').mean()
            v.close()
            pipe = os.popen("mincstats -mean -quiet %s" % inputFile_float, "r")
            output = float(pipe.read())
            pipe.close()
            self.assertAlmostEqual(a, output, 8)
        def testFromFileDataDouble(self):
            """ensure that double data is read correct with a precision of 8 decimals on a call to aveage()"""
            v = minc2_file(inputFile_double) 
            a = v.tensor.mean()
            v.close()
            pipe = os.popen("mincstats -mean -quiet %s" % inputFile_double, "r")
            output = float(pipe.read())
            pipe.close()
            self.assertAlmostEqual(a, output, 8)
except ImportError:
    pass

class TestWriteFileDataTypes(unittest.TestCase):
    ############################################################################
    # volumeFromDescription
    ############################################################################
    def testWriteDataAsByte(self):
        """ensure that a volume created by volumeFromDescription as byte is written out as such"""
        # TODO
        pass
    def testWriteDataAsShort(self):
        """ensure that a volume created by volumeFromDescription as short is written out as such"""
        # TODO
        pass
    def testWriteDataAsInt(self):
        """ensure that a volume created by volumeFromDescription as int is written out as such"""
        # TODO
        pass
    def testWriteDataAsFloat(self):
        """ensure that a volume created by volumeFromDescription as float is written out as such"""
        # TODO
        pass
    def testWriteDataAsDouble(self):
        """ensure that a volume created by volumeFromDescription as double is written out as such"""
        # TODO
        pass
    def testWriteDataAsUByte(self):
        """ensure that a volume created by volumeFromDescription as unsigned byte is written out as such"""
        # TODO
        pass
    def testWriteDataAsUShort(self):
        """ensure that a volume created by volumeFromDescription as unsigned short is written out as such"""
        # TODO
        pass
    def testWriteDataAsUInt(self):
        """ensure that a volume created by volumeFromDescription as unsigned int is written out as such"""
        # TODO
        pass

class TestHyperslabs(unittest.TestCase):
    """test getting and setting of hyperslabs"""
    def testGetHyperslab(self):
        """hyperslab should be same as slice from data array"""
        
        inputFile=inputFile_ushort
        #inputFile='/export01/data/vfonov/src1/minc2-simple/python/test_icbm.mnc'
        
        v = minc2_file(inputFile)
        v.setup_standard_order()
        sliceFromData_x = v.data[10,:,:]
        sliceFromData_y = v.data[:,10,:]
        sliceFromData_z = v.data[:,:,10]
        v.close()
        
        b = minc2_file(inputFile)
        b.setup_standard_order()
        hyperslab_x = b.load_hyperslab( [10, None, None] ).squeeze()
        hyperslab_y = b.load_hyperslab( [None, 10, None] ).squeeze()
        hyperslab_z = b.load_hyperslab( [None, None, 10] ).squeeze()
        b.close()

        self.assertEqual(N.average((sliceFromData_x-hyperslab_x)**2),0.0)
        self.assertEqual(N.average((sliceFromData_y-hyperslab_y)**2),0.0)
        self.assertEqual(N.average((sliceFromData_z-hyperslab_z)**2),0.0)
        
    def testSetHyperslabFloat(self):
        """setting hyperslab should change underlying volume (float)"""
        
        # read some data from somwhere
        v  = minc2_file(inputFile_ushort)
        dims=v.store_dims()
        v.setup_standard_order()
        hyperslab_a = v.load_hyperslab( [10, None, None] )
        
        v2 = minc2_file()
        v2.define(dims,'float32','float32')
        v2.create(outputFilename)
        v2.setup_standard_order()
        
        # because we are saving float32 , we don't need slice normalization
        v2.save_hyperslab(hyperslab_a,   [10,None,None] )
        hyperslab_b = v2.load_hyperslab( [10, None, None] )
        self.assertEqual(N.average((hyperslab_a-hyperslab_b)**2),0.0)
        v2.close()
        v.close()

    def testSetHyperslabShort(self):
        """setting hyperslab should change underlying volume (short)"""
        
        # read some data from somwhere
        v  = minc2_file(inputFile_ushort)
        dims=v.store_dims()
        v.setup_standard_order()
        hyperslab_a = v.load_hyperslab( [10, None, None] )
        
        # try with normalization
        v2 = minc2_file()
        v2.define(dims,'uint16','float32') # , global_scaling=True
        v2.create(outputFilename)
        v2.set_volume_range(N.min(hyperslab_a),N.max(hyperslab_a))
        v2.setup_standard_order()
        
        # have to set slice normalization
        v2.save_hyperslab(hyperslab_a,   [10,None,None] )
        hyperslab_b = v2.load_hyperslab( [10, None, None] )
        self.assertAlmostEqual(N.average((hyperslab_a-hyperslab_b)**2),0.0,8)
        v2.close()
        v.close()
        
        
    def testHyperslabArray(self):
        """hyperslab should be reinsertable into volume"""
        if False:
            v = minc2_file(inputFile_ushort)
            v2 = minc2_file()
            v2.create(outputFilename)
            v2.close()
            v.close()

try: # run tests if torch is present
    import torch
    
    class TestHyperslabsTensors(unittest.TestCase):
        """test getting and setting of hyperslabs"""
        def testGetHyperslab(self):
            """hyperslab should be same as slice from data array"""
            
            inputFile=inputFile_ushort
            #inputFile='/export01/data/vfonov/src1/minc2-simple/python/test_icbm.mnc'
            
            v = minc2_file(inputFile)
            v.setup_standard_order()
            sliceFromData_x = v.data[10,:,:]
            sliceFromData_y = v.data[:,10,:]
            sliceFromData_z = v.data[:,:,10]
            v.close()
            
            b = minc2_file(inputFile)
            b.setup_standard_order()
            hyperslab_x = b.load_hyperslab_t( [10, None, None] ).squeeze()
            hyperslab_y = b.load_hyperslab_t( [None, 10, None] ).squeeze()
            hyperslab_z = b.load_hyperslab_t( [None, None, 10] ).squeeze()
            b.close()

            self.assertEqual(torch.mean((sliceFromData_x-hyperslab_x)**2),0.0)
            self.assertEqual(torch.mean((sliceFromData_y-hyperslab_y)**2),0.0)
            self.assertEqual(torch.mean((sliceFromData_z-hyperslab_z)**2),0.0)
            
        def testSetHyperslabFloat(self):
            """setting hyperslab should change underlying volume (float)"""
            
            # read some data from somwhere
            v  = minc2_file(inputFile_ushort)
            dims=v.store_dims()
            v.setup_standard_order()
            hyperslab_a = v.load_hyperslab_t( [10, None, None] )
            
            v2 = minc2_file()
            v2.define(dims,'float32','float32')
            v2.create(outputFilename)
            v2.setup_standard_order()
            
            # because we are saving float32 , we don't need slice normalization
            v2.save_hyperslab_t(hyperslab_a,   [10,None,None] )
            hyperslab_b = v2.load_hyperslab_t( [10, None, None] )
            self.assertEqual(N.average((hyperslab_a-hyperslab_b)**2),0.0)
            v2.close()
            v.close()

        def testSetHyperslabShort(self):
            """setting hyperslab should change underlying volume (short)"""
            
            # read some data from somwhere
            v  = minc2_file(inputFile_ushort)
            dims=v.store_dims()
            v.setup_standard_order()
            hyperslab_a = v.load_hyperslab_t( [10, None, None] )
            
            # try with normalization
            v2 = minc2_file()
            v2.define(dims,'uint16','float32') # , global_scaling=True
            v2.create(outputFilename)
            v2.set_volume_range(torch.min(hyperslab_a),torch.max(hyperslab_a))
            v2.setup_standard_order()
            
            # have to set slice normalization
            v2.save_hyperslab_t(hyperslab_a,   [10,None,None] )
            hyperslab_b = v2.load_hyperslab_t( [10, None, None] )
            self.assertAlmostEqual(torch.mean((hyperslab_a-hyperslab_b)**2),0.0,8)
            v2.close()
            v.close()
            
            
        def testHyperslabArray(self):
            """hyperslab should be reinsertable into volume"""
            if False:
                v = minc2_file(inputFile_ushort)
                v2 = minc2_file()
                v2.create(outputFilename)
                v2.close()
                v.close()
                
except ImportError:
    pass


class testVectorFiles(unittest.TestCase):
    """test reading and writing of vector files"""
    def testVectorRead(self):
        """make sure that a vector file can be read correctly"""
        v = minc2_file(inputVector)
        v.setup_standard_order()
        dims = v.representation_dims()
        self.assertEqual(dims[0].id, minc2_file.MINC2_DIM_VEC)
        v.close()
    def testVectorRead2(self):
        """make sure that volume has four dimensions"""
        v = minc2_file(inputVector)
        ndims = v.ndim()
        self.assertEqual(ndims, 4)
        data=v.data
        self.assertEqual(len(data.shape),4)
        v.close()
        
class testDirectionCosines(unittest.TestCase):
    """test that pyminc deals correctly with direction cosines"""
    def testDefaultDirCos3DVFF(self):
        """testing reading the direction cosines of a file with standard values (volumeFromFile)"""
        v = minc2_file(inputFile_ushort)
        #
        # This file was created without explicitly setting the direction cosines.
        # in that case, the attribute is not set altogether, so we should test
        # for it using the known defaults, because libminc does extract the correct
        # default values
        #
        v.setup_standard_order()
        dims = v.representation_dims()
        self.assertAlmostEqual(dims[0].dir_cos[0], 1.0, 8)
        self.assertAlmostEqual(dims[0].dir_cos[1], 0.0, 8)
        self.assertAlmostEqual(dims[0].dir_cos[2], 0.0, 8)
        
        self.assertAlmostEqual(dims[1].dir_cos[0], 0.0, 8)
        self.assertAlmostEqual(dims[1].dir_cos[1], 1.0, 8)
        self.assertAlmostEqual(dims[1].dir_cos[2], 0.0, 8)
        
        self.assertAlmostEqual(dims[2].dir_cos[0], 0.0, 8)
        self.assertAlmostEqual(dims[2].dir_cos[1], 0.0, 8)
        self.assertAlmostEqual(dims[2].dir_cos[2], 1.0, 8)
        
    def testNonDefaultDirCos3DVFF(self):
        """testing reading the direction cosines of a file with non-standard values (volumeFromFile)"""
        v = minc2_file(input3DdirectionCosines)
        v.setup_standard_order()
        dims = v.representation_dims()
        
        pipe = os.popen("mincinfo -attvalue xspace:direction_cosines %s" % input3DdirectionCosines, "r")
        from_file = pipe.read().rstrip().split(" ")
        pipe.close()
        
        self.assertAlmostEqual(dims[0].dir_cos[0], float(from_file[0]), 8)
        self.assertAlmostEqual(dims[0].dir_cos[1], float(from_file[1]), 8)
        self.assertAlmostEqual(dims[0].dir_cos[2], float(from_file[2]), 8)
        
        pipe = os.popen("mincinfo -attvalue yspace:direction_cosines %s" % input3DdirectionCosines, "r")
        from_file = pipe.read().rstrip().split(" ")
        pipe.close()
        self.assertAlmostEqual(dims[1].dir_cos[0], float(from_file[0]), 8)
        self.assertAlmostEqual(dims[1].dir_cos[1], float(from_file[1]), 8)
        self.assertAlmostEqual(dims[1].dir_cos[2], float(from_file[2]), 8)
        
        pipe = os.popen("mincinfo -attvalue zspace:direction_cosines %s" % input3DdirectionCosines, "r")
        from_file = pipe.read().rstrip().split(" ")
        pipe.close()
        self.assertAlmostEqual(dims[2].dir_cos[0], float(from_file[0]), 8)
        self.assertAlmostEqual(dims[2].dir_cos[1], float(from_file[1]), 8)
        self.assertAlmostEqual(dims[2].dir_cos[2], float(from_file[2]), 8)
        

class testXfmsAppliedToCoordinates(unittest.TestCase):
    """test that xfm files can be used to transform x,y,z coordinates"""
    def testForwardTransformSingleXfm(self):
        """testing coordinates transformed using the forward transform and a single transformation"""
        _xfm=minc2_xfm(outputXfmFilename1)
        out=_xfm.transform_point([6.68, 3.14, 7.00])
        self.assertAlmostEqual(out[0], 4.33400016486645, 8)
        self.assertAlmostEqual(out[1], 32.3265016365052, 8)
        self.assertAlmostEqual(out[2], -12.4399995803833, 8)
    
    def testInverseTransformSingleXfm(self):
        """testing coordinates transformed using the inverse transform and a single transformation"""
        
        _xfm=minc2_xfm(outputXfmFilename1)
        out=_xfm.inverse_transform_point([6.68, 3.14, 7.00])
        self.assertAlmostEqual(out[0], 18.4099990008772, 8)
        self.assertAlmostEqual(out[1], -3.64755821904214, 8)
        self.assertAlmostEqual(out[2], 0.520000139872233, 8)
    
    def testForwardTransformConcatenatedXfm(self):
        """testing coordinates transformed using the forward transform and a concatenated transformation"""
        
        _xfm=minc2_xfm(outputXfmFilename3)
        out=_xfm.transform_point([6.68, 3.14, 7.00])
        self.assertAlmostEqual(out[0],  259.159993714094, 8)
        self.assertAlmostEqual(out[1],  188.041454144745, 8)
        self.assertAlmostEqual(out[2], -1744.15997695923, 8)
    
    def testInverseTransformConcatenatedXfm(self):
        """testing coordinates transformed using the inverse transform and a concatenated transformation"""
        
        _xfm=minc2_xfm(outputXfmFilename3)
        out=_xfm.inverse_transform_point([6.68, 3.14, 7.00])
        self.assertAlmostEqual(out[0], -119.559994975925, 8)
        self.assertAlmostEqual(out[1], -2.72634880128239, 8)
        self.assertAlmostEqual(out[2], 0.0509524723840147, 8)
    
        
        
if __name__ == "__main__":
    unittest.main()
