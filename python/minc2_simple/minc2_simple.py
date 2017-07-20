from ._simple import ffi,lib
from .utils   import to_bytes
from .utils   import text_type
import six

class minc2_error(Exception):
    pass

class minc2_transform_parameters(object):
    def __init__(self):
        import numpy as np
        self.center=np.zeros(3)
        self.translations=np.zeros(3)
        self.scales=np.zeros(3)
        self.shears=np.zeros(3)
        self.rotations=np.zeros(3)
        self.invalid=False

    def __str__(self):
        return "center: {} {} {}\n".format(self.center[0],self.center[1],self.center[2])+ \
               "translations: {} {} {}\n".format(self.translations[0],self.translations[1],self.translations[2])+ \
               "scales: {} {} {}\n".format(self.scales[0],self.scales[1],self.scales[2])+ \
               "rotations: {} {} {}\n".format(self.rotations[0],self.rotations[1],self.rotations[2])+ \
               "shears: {} {} {}\n".format(self.shears[0],self.shears[1],self.shears[2])

    def __repr__(self):
        return self.__str__()

class minc2_file(object):

    # constants
    MINC2_DIM_UNKNOWN=lib.MINC2_DIM_UNKNOWN
    MINC2_DIM_X    = lib.MINC2_DIM_X
    MINC2_DIM_Y    = lib.MINC2_DIM_Y
    MINC2_DIM_Z    = lib.MINC2_DIM_Z
    MINC2_DIM_TIME = lib.MINC2_DIM_TIME
    MINC2_DIM_VEC  = lib.MINC2_DIM_VEC
    MINC2_DIM_END  = lib.MINC2_DIM_END
    
    # minc2 data types
    MINC2_BYTE     = lib.MINC2_BYTE 
    MINC2_SHORT    = lib.MINC2_SHORT
    MINC2_INT      = lib.MINC2_INT 
    MINC2_FLOAT    = lib.MINC2_FLOAT 
    MINC2_DOUBLE   = lib.MINC2_DOUBLE 
    MINC2_STRING   = lib.MINC2_STRING 
    MINC2_UBYTE    = lib.MINC2_UBYTE 
    MINC2_USHORT   = lib.MINC2_USHORT 
    MINC2_UINT     = lib.MINC2_UINT 
    MINC2_SCOMPLEX = lib.MINC2_SCOMPLEX 
    MINC2_ICOMPLEX = lib.MINC2_ICOMPLEX 
    MINC2_FCOMPLEX = lib.MINC2_FCOMPLEX 
    MINC2_DCOMPLEX = lib.MINC2_DCOMPLEX 
    MINC2_MAX_TYPE_ID=lib.MINC2_MAX_TYPE_ID
    MINC2_UNKNOWN  = lib.MINC2_UNKNOWN  

    # minc2 status
    MINC2_SUCCESS  = lib.MINC2_SUCCESS,
    MINC2_ERROR    = lib.MINC2_ERROR

    # data types
    __minc2_to_numpy={
            lib.MINC2_BYTE :   'int8',
            lib.MINC2_UBYTE :  'uint8',
            lib.MINC2_SHORT :  'int16',
            lib.MINC2_USHORT : 'uint16',
            lib.MINC2_INT :    'int32',
            lib.MINC2_UINT :   'uint32',
            lib.MINC2_FLOAT :  'float32',
            lib.MINC2_DOUBLE : 'float64',
        }

    __numpy_to_minc2 = {y:x for x,y in six.iteritems(__minc2_to_numpy)}
    
    def __init__(self, path=None):
        self._v=ffi.gc(lib.minc2_allocate0(),lib.minc2_destroy)
        if path is not None:
            self.open(path)
        
    def open(self, path):
        if lib.minc2_open(self._v, to_bytes(path))!=lib.MINC2_SUCCESS:
            raise minc2_error()
        
    def close(self):
        if lib.minc2_close(self._v)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        
    def ndim(self):
        dd=ffi.new("int*", 0)
        lib.minc2_ndim(self._v,dd)
        return dd[0]
    
    def store_dims(self):
        dims=ffi.new("struct minc2_dimension*[1]")
        if lib.minc2_get_store_dimensions(self._v,dims)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return dims[0]
    
    def representation_dims(self):
        dims=ffi.new("struct minc2_dimension*[1]")
        if lib.minc2_get_representation_dimensions(self._v,dims)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return dims[0]
    
    def define(self, dims, store_type=None, representation_type=None):
        _dims=dims
        if store_type is None:
            store_type=lib.MINC2_SHORT
        if representation_type is None:
            representation_type=store_type

        _store_type=store_type
        _representation_type=representation_type

        if isinstance(_store_type, six.string_types):
            _store_type=__numpy_to_minc2[_store_type]

        if isinstance(_representation_type, six.string_types):
            _representation_type=__numpy_to_minc2[_representation_type]

        if isinstance(dims,list):
            _dims = ffi.new("struct minc2_dimension[]", len(dims)+1)
            for i,j in enumerate(dims):
                _dims[i]=j
            _dims[len(dims)]={'id':lib.MINC2_DIM_END}

        if lib.minc2_define(self._v, _dims, _store_type, _representation_type)!=lib.MINC2_SUCCESS:
            raise minc2_error()
    
    def create(self, path):
        if lib.minc2_create(self._v, to_bytes(path) )!=lib.MINC2_SUCCESS:
            raise minc2_error()
    
    def copy_metadata(self, another):
        if lib.minc2_copy_metadata(another._v,self._v)!=lib.MINC2_SUCCESS:
            raise minc2_error()
    
    def load_complete_volume(self, data_type=None):
        import numpy as np
        if data_type is None:
            data_type=self.representation_dtype()
        buf=None
        _dims=self.representation_dims()
        # dims=torch.LongStorage(self:ndim())
        shape=list(range(self.ndim()))
        # numpy array  defines dimensions in a slowest first fashion
        for i in range(self.ndim()):
            shape[self.ndim()-i-1]=_dims[i].length

        dtype=None
        
        if data_type in minc2_file.__minc2_to_numpy:
            dtype=minc2_file.__minc2_to_numpy[data_type]
        elif data_type in minc2_file.__numpy_to_minc2:
            dtype=data_type
            data_type=minc2_file.__numpy_to_minc2[dtype]
        elif isinstance(data_type,np.dtype):
            dtype=data_type
            data_type=minc2_file.__numpy_to_minc2[dtype.name]
        else:
            raise minc2_error()
        buf=np.empty(shape,dtype,'C')
        if lib.minc2_load_complete_volume(self._v, ffi.cast("void *", buf.ctypes.data) , data_type)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return buf

    def setup_standard_order(self):
        if lib.minc2_setup_standard_order(self._v)!=lib.MINC2_SUCCESS:
            raise minc2_error()
    
    def save_complete_volume(self, buf):
        import numpy as np
        data_type=lib.MINC2_FLOAT
        store_type=buf.dtype.name
        # TODO: make sure array is in "C" order
        
        assert(store_type in minc2_file.__numpy_to_minc2)
        # TODO: verify dimensions of the array

        data_type=minc2_file.__numpy_to_minc2[store_type]
        
        if lib.minc2_save_complete_volume(self._v,ffi.cast("void *", buf.ctypes.data),data_type)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return buf

    def read_attribute(self,group,attribute):
        import numpy as np

        attr_type=ffi.new("int*",0)
        attr_length=ffi.new("int*",0)

        group=to_bytes(group)
        attribute=to_bytes(attribute)
        # assume that if we can't get attribute type, it's missing, return nil

        if lib.minc2_get_attribute_type(self._v,group,attribute,attr_type)!=lib.MINC2_SUCCESS:
            return None

        if lib.minc2_get_attribute_length(self._v,group,attribute,attr_length)!=lib.MINC2_SUCCESS:
            raise minc2_error()

        if attr_type[0] == lib.MINC2_STRING:
            buf = ffi.new("char[]", attr_length[0])
            if lib.minc2_read_attribute(self._v,group,attribute,buf,attr_length[0])!=lib.MINC2_SUCCESS:
                raise minc2_error()
            return ffi.string(buf, attr_length[0])
        else:
            data_type=attr_type[0]
            buf=None
            if data_type in minc2_file.__minc2_to_numpy:
                dtype=minc2_file.__minc2_to_numpy[data_type]
                shape=[attr_length[0]]
                buf=np.empty(shape,dtype,'C')
            else:
                raise minc2_error()

            if lib.minc2_read_attribute(self._v,group,attribute,ffi.cast("void *", buf.ctypes.data),attr_length[0])!=lib.MINC2_SUCCESS:
                raise minc2_error()

            return buf

    def write_attribute(self,group,attribute,value):
        group=to_bytes(group)
        attribute=to_bytes(attribute)

        if isinstance(value, text_type):
            attr_type=lib.MINC2_STRING
            attr_length=len(value)

            if lib.minc2_write_attribute(self._v,group,attribute,ffi.cast("const char[]",to_bytes(value)),attr_length+1,lib.MINC2_STRING)!=lib.MINC2_SUCCESS:
                raise minc2_error()
        else:
            import numpy as np
            data_type=lib.MINC2_FLOAT
            store_type=buf.dtype.name

            assert(store_type in minc2_file.__numpy_to_minc2)
            data_type=minc2_file.__numpy_to_minc2[store_type]

            if lib.minc2_write_attribute(self._v,group,attribute,ffi.cast("void *", buf.ctypes.data),buf.size,data_type)!=lib.MINC2_SUCCESS:
                raise minc2_error()

    def metadata(self):
        ret={}

        group_iterator=ffi.gc(lib.minc2_allocate_info_iterator(), lib.minc2_free_info_iterator)
        attr_iterator=ffi.gc(lib.minc2_allocate_info_iterator(), lib.minc2_free_info_iterator)

        if lib.minc2_start_group_iterator(self._v,group_iterator)!=lib.MINC2_SUCCESS:
            raise minc2_error()

        while lib.minc2_iterator_group_next(group_iterator)==lib.MINC2_SUCCESS:
            gname=lib.minc2_iterator_group_name(group_iterator)
            if lib.minc2_start_attribute_iterator(self._v, gname, attr_iterator)!=lib.MINC2_SUCCESS:
                raise minc2_error()
            g={}

            while lib.minc2_iterator_attribute_next(attr_iterator)==lib.MINC2_SUCCESS:
                aname=lib.minc2_iterator_attribute_name(attr_iterator)
                g[ ffi.string(aname) ] = self.read_attribute(gname, aname)

            ret[ ffi.string(lib.minc2_iterator_group_name(group_iterator)) ] = g
            lib.minc2_stop_info_iterator(attr_iterator)

        lib.minc2_stop_info_iterator(group_iterator)
        return ret

    def write_metadata(self,m):
        for group,g in six.iteritems(m):
            for attr,a in six.iteritems(g):
                self.write_attribute(group,attr,a)

    def store_dtype(self):
        _dtype=ffi.new("int*",0)
        if lib.minc2_storage_data_type(self._v,_dtype)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return minc2_file.__minc2_to_numpy[_dtype[0]]

    def representation_dtype(self):
        _dtype=ffi.new("int*",0)
        if lib.minc2_data_type(self._v,_dtype)!=lib.MINC2_SUCCESS:
            raise minc2_error()
        return minc2_file.__minc2_to_numpy[_dtype[0]]

    # shortcut to setup_standard_order;load_complete_volume
    def get_data(self):
        self.setup_standard_order()
        return self.load_complete_volume()

    # shortcut to setup_standard_order,save_complete_volume
    def set_data(self,new_data):
        self.setup_standard_order()
        self.save_complete_volume(new_data)

    # get/set whole volume
    data = property(get_data,set_data,None,"Complete volume")

    # hyperslab-based functions
    def set_volume_range(self, rmin, rmax):
        if lib.minc2_set_volume_range(self._v,rmin,rmax) != lib.MINC2_SUCCESS:
            raise minc2_error()

    def save_hyperslab(self, buf, start=None):
        if start is None:
            return self.save_complete_volume(buf)
        else:
            import numpy as np
            data_type=lib.MINC2_FLOAT
            store_type=buf.dtype.name

            ndims =self.ndim()
            dims = buf.shape

            slab_start=ffi.new("int[]",ndims)
            slab_count=ffi.new("int[]",ndims)

            for i in range(ndims):
                if start[i] is not None:
                    if isinstance(start[i],list):
                        if len(start[i])==2:
                            slab_count[ndims-1-i]=start[i][1]-start[i][0]
                            slab_start[ndims-1-i]=start[i][0]
                        else: # assume it's the whole dimension
                            slab_count[ndims-1-i]=dims[i]
                            slab_start[ndims-1-i]=0
                    else: #assume it's a number
                        slab_count[ndims-1-i]=dims[i]
                        slab_start[ndims-1-i]=start[i]
                else:
                    slab_count[ndims-1-i]=dims[i]
                    slab_start[ndims-1-i]=0

            data_type=minc2_file.__numpy_to_minc2[store_type]

            if lib.minc2_write_hyperslab(self._v, slab_start, slab_count,
                                         ffi.cast("void *", buf.ctypes.data), data_type)!=lib.MINC2_SUCCESS:
                raise minc2_error()
            return buf

    def load_hyperslab(self, slab=None, data_type=None):
        if data_type is None:
            data_type=self.representation_dtype()
        if slab is None:
            return self.load_complete_volume(data_type)
        else:
            import numpy as np
            buf = None
            _dims = self.representation_dims()
            ndims = self.ndim()
            dims = [None]*ndims

            slab_start = ffi.new("int[]",ndims)
            slab_count = ffi.new("int[]",ndims)

            for i in range(ndims):
                if slab[i] is not None:
                    if isinstance(slab[i], list):
                        if len(slab[i]) == 2:
                            slab_count[ndims-1-i] = slab[i][1]-slab[i][0]
                            slab_start[ndims-1-i] = slab[i][0]
                        else:# -- assume it's the whole dimension
                            slab_count[ndims-1-i] = _dims[ndims-i-1].length
                            slab_start[ndims-1-i] = 0
                    else: # assume it's a number
                        slab_count[ndims-1-i] = 1
                        slab_start[ndims-1-i] = slab[i]
                else:
                    slab_count[ndims-1-i] = _dims[ndims-i-1].length
                    slab_start[ndims-1-i] = 0
                dims[i] = slab_count[ndims-1-i]

            dtype = None

            if data_type in minc2_file.__minc2_to_numpy:
                dtype=minc2_file.__minc2_to_numpy[data_type]
            elif data_type in minc2_file.__numpy_to_minc2:
                dtype=data_type
                data_type=minc2_file.__numpy_to_minc2[dtype]
            elif isinstance(data_type, np.dtype):
                dtype=data_type
                data_type=minc2_file.__numpy_to_minc2[dtype.name]
            else:
                raise minc2_error()

            buf = np.empty(dims, dtype, 'C')

            if lib.minc2_read_hyperslab(self._v, slab_start, slab_count,
                                        ffi.cast("void *", buf.ctypes.data), data_type) != lib.MINC2_SUCCESS:
                raise minc2_error()
            return buf


class minc2_xfm(object):
    # constants
    MINC2_XFM_LINEAR                 = lib.MINC2_XFM_LINEAR
    MINC2_XFM_THIN_PLATE_SPLINE      = lib.MINC2_XFM_THIN_PLATE_SPLINE
    MINC2_XFM_USER_TRANSFORM         = lib.MINC2_XFM_USER_TRANSFORM
    MINC2_XFM_CONCATENATED_TRANSFORM = lib.MINC2_XFM_CONCATENATED_TRANSFORM
    MINC2_XFM_GRID_TRANSFORM         = lib.MINC2_XFM_GRID_TRANSFORM


    def __init__(self,path=None):
        self._v=ffi.gc(lib.minc2_xfm_allocate0(),lib.minc2_xfm_destroy)
        if path is not None:
            self.open(path)


    def open(self,path):
        assert path is not None,"Provide minc2 file"
        assert lib.minc2_xfm_open(self._v,path) == lib.MINC2_SUCCESS


    def save(self,path):
        assert path is not None,"Provide minc2 file"
        assert(lib.minc2_xfm_save(self._v,path) == lib.MINC2_SUCCESS)


    def transform_point(self,xyz_in):
        import numpy as np
        _xyz_in=np.asarray(xyz_in,'float64','C')
        xyz_out=np.empty([3],'float64','C')
        assert(lib.minc2_xfm_transform_point(self._v,ffi.cast("double *", _xyz_in.ctypes.data),ffi.cast("double *", xyz_out.ctypes.data))==lib.MINC2_SUCCESS)
        return xyz_out


    def inverse_transform_point(self,xyz_in):
        import numpy as np
        _xyz_in=np.asarray(xyz_in,'float64','C')
        xyz_out=np.empty([3],'float64','C')
        assert(lib.minc2_xfm_inverse_transform_point(self._v,ffi.cast("double *", _xyz_in.ctypes.data),ffi.cast("double *", xyz_out.ctypes.data))==lib.MINC2_SUCCESS)
        return xyz_out


    def invert(self):
        assert(lib.minc2_xfm_invert(self._v)==lib.MINC2_SUCCESS)

    def get_n_concat(self):
        n=ffi.new("int*",0)
        assert(lib.minc2_xfm_get_n_concat(self._v,n)==lib.MINC2_SUCCESS)
        return n[0]

    def get_n_type(self,n=0):
        t=ffi.new("int*",0)
        assert(lib.minc2_xfm_get_n_type(self._v,n,t)==lib.MINC2_SUCCESS)
        return t[0]

    def get_grid_transform(self,n=0):
        warnings.warn("Grid transform is not tested")
        c_file=ffi.new("char**")
        inv=ffi.new("int*",0)
        assert(lib.minc2_xfm_get_grid_transform(self._v,n,inv,c_file)==lib.MINC2_SUCCESS)
        _file=ffi.string(c_file[0])
        ffi.free(c_file[0])
        return (_file,inv[0])

    def get_linear_transform(self,n=0):
        import numpy as np
        _mat=np.empty([4,4],'float64','C')
        assert(lib.minc2_xfm_get_linear_transform(self._v,n,ffi.cast("double *", _mat.ctypes.data))==lib.MINC2_SUCCESS)
        return _mat

    def get_linear_transform_param(self,n=0,center=None):
        import numpy as np
        par=minc2_transform_parameters()

        if center is not None:
            par.center=np.asarray(center,'float64','C')
        # TODO: detect if extraction of parameters failed

        if lib.minc2_xfm_extract_linear_param(self._v,n,
                ffi.cast("double *", par.center.ctypes.data),
                ffi.cast("double *", par.translations.ctypes.data),
                ffi.cast("double *", par.scales.ctypes.data),
                ffi.cast("double *", par.shears.ctypes.data),
                ffi.cast("double *", par.rotations.ctypes.data)
            )!=lib.MINC2_SUCCESS:
            par.invalid=True
        return par


    def append_linear_transform(self,par):
        import numpy as np
        if isinstance(par,np.ndarray): # assume it's a matrix
            _mat=np.asarray(par,'float64','C')
            assert(lib.minc2_xfm_append_linear_transform(self._v,ffi.cast("double *", _mat.ctypes.data))==lib.MINC2_SUCCESS)
        else: # must be an object with parameters
            _par=minc2_transform_parameters()

            _par.center=np.asarray(par.center,'float64','C')
            _par.translations=np.asarray(par.translations,'float64','C')
            _par.scales=np.asarray(par.scales,'float64','C')
            _par.shears=np.asarray(par.shears,'float64','C')
            _par.rotations=np.asarray(par.rotations,'float64','C')

            assert(lib.minc2_xfm_append_linear_param(self._v,
                ffi.cast("double *", _par.center.ctypes.data),
                ffi.cast("double *", _par.translations.ctypes.data),
                ffi.cast("double *", _par.scales.ctypes.data),
                ffi.cast("double *", _par.shears.ctypes.data),
                ffi.cast("double *", _par.rotations.ctypes.data)
                )==lib.MINC2_SUCCESS)
        return self

    def append_grid_transform(self,grid_file,inv=False):
        warnings.warn("Grid transform is not tested")
        assert(lib.minc2_xfm_append_grid_transform(self._v,grid_file,inv)==lib.MINC2_SUCCESS)
        return self

    def concat_xfm(self,another):
        assert(lib.minc2_xfm_concat_xfm(self._v,another._v)==lib.MINC2_SUCCESS)

# kate: indent-width 4; replace-tabs on; remove-trailing-space on; hl python; show-tabs on
