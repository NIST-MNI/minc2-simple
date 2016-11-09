from ._simple import ffi,lib
from .utils   import to_bytes
from .utils   import text_type


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
    
    def __init__(self,path=None):
        self._v=ffi.gc(lib.minc2_allocate0(),lib.minc2_destroy)
        if path is not None:
            lib.minc2_open(self._v,to_bytes(path))
        
    def open(self,path):
        lib.minc2_open(self._v,to_bytes(path))
        
    def close(self):
        lib.minc2_close(self._v)
        
    def ndim(self):
        dd=ffi.new("int*", 0)
        lib.minc2_ndim(self._v,dd)
        return dd[0]
    
    def store_dims(self):
        dims=ffi.new("struct minc2_dimension*[1]")
        lib.minc2_get_store_dimensions(self._v,dims)
        return dims[0]
    
    def representation_dims(self):
        dims=ffi.new("struct minc2_dimension*[1]")
        lib.minc2_get_representation_dimensions(self._v,dims)
        return dims[0]
    
    def define(self,dims,store_type,representation_type):
        _dims=dims
        if isinstance(dims,list):
            _dims = ffi.new("struct minc2_dimension[]", len(dims)+1)
            for i,j in enumerate(dims):
                _dims[i]=j
            _dims[len(dims)]={'id':lib.MINC2_DIM_END}
        
        lib.minc2_define(self._v,_dims,store_type,representation_type)
    
    
    def create(self,path):
        lib.minc2_create(self._v, path )
    
    def copy_metadata(self,another):
        lib.minc2_copy_metadata(another._v,self._v)
    
    def load_complete_volume(self,data_type):
        import numpy as np
        
        buf=None
        _dims=self.representation_dims()
        # dims=torch.LongStorage(self:ndim())
        shape=range(self.ndim())
        # numpy array  defines dimensions in a slowest first fashion
        for i in range(self.ndim()):
            shape[self.ndim()-i-1]=_dims[i].length
        dtype='float'
        if data_type==lib.MINC2_BYTE : 
            dtype='int8'
        elif data_type==lib.MINC2_UBYTE : 
            dtype='uint8'
        elif data_type==lib.MINC2_SHORT : 
            dtype='int16'
        elif data_type==lib.MINC2_USHORT : 
            dtype='uint16'
        elif data_type==lib.MINC2_INT : 
            dtype='int32'
        elif data_type==lib.MINC2_UINT : 
            dtype='uint32'
        elif data_type==lib.MINC2_FLOAT : 
            dtype='float32'
        elif data_type==lib.MINC2_DOUBLE : 
            dtype='float64'
        else :
            # unsupported
            pass
        buf=np.empty(shape,dtype,'C')
        lib.minc2_load_complete_volume(self._v, ffi.cast("void *", buf.ctypes.data) , data_type)#
        
        return buf
    
    def setup_standard_order(self):
        lib.minc2_setup_standard_order(self._v)
    
    def save_complete_volume(self,buf):
        import numpy as np
        data_type=lib.MINC2_FLOAT
        store_type=buf.dtype
        # TODO: make sure array is in "C" order
        
        if store_type == np.dtype('int8'):
            data_type=lib.MINC2_BYTE
        elif store_type==np.dtype('uint8') : 
            data_type=lib.MINC2_UBYTE
        elif store_type==np.dtype('int16') : 
            data_type=lib.MINC2_SHORT
        elif store_type==np.dtype('uint8') : 
            data_type=lib.MINC2_USHORT
        elif store_type==np.dtype('int32') : 
            data_type=lib.MINC2_INT
        elif store_type == np.dtype('uint32') : 
            data_type=lib.MINC2_UINT
        elif store_type == np.dtype('float32') : 
            data_type=lib.MINC2_FLOAT
        elif store_type == np.dtype('float64') : 
            data_type=lib.MINC2_DOUBLE
        else:
            # not supported
            pass
        
        lib.minc2_save_complete_volume(self._v,ffi.cast("void *", buf.ctypes.data),data_type)
        
        return buf
