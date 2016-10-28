from m2_simple import ffi,lib
from utils import to_bytes
from utils import text_type



class minc2_file(object):
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
    


if __name__ == "__main__":
    m=minc2_file("/extra/mni/me/mc_fonv7706.2007-06-14_09-42-45.Z25-03_S_nrx-t1g.mnc.gz")
    print("dims={}".format(m.ndim()))
    m.close()
    