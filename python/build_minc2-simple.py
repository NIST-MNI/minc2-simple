import os
import sys
from cffi import FFI

ffi = FFI()

print(os.path.dirname(__file__))
src=""
with open(os.path.join(os.path.dirname(__file__), "../src/minc2-simple.c"),'r') as f:
    src=f.read()

ffi.set_source("m2_simple",
    src,
    # The important thing is to include libc in the list of libraries we're
    # linking against:
    libraries=["minc2","c"],
    include_dirs=["/opt/minc/1.9.13/include","/home/vfonov/src/minc2-simple/src"],
    library_dirs=["/home/vfonov/src/build/minc2-simple/src","/opt/minc/1.9.13/lib"],
    
)

ffi.cdef(
    """
/**
  * minc2 dimension types
  */ 
enum  minc2_dimensions {
  MINC2_DIM_UNKNOWN=0,
  MINC2_DIM_X,
  MINC2_DIM_Y,
  MINC2_DIM_Z,
  MINC2_DIM_TIME,
  MINC2_DIM_VEC,
  MINC2_DIM_END=255
};

/**
 * minc2 data storage type
 * compatible with minc2 API
 */
enum  minc2_type {
  MINC2_ORIGINAL = 0,     /**< MI_ORIGINAL_TYPE */
  MINC2_BYTE = 1,         /**< 8-bit signed integer */
  MINC2_SHORT = 3,        /**< 16-bit signed integer */
  MINC2_INT = 4,          /**< 32-bit signed integer */
  MINC2_FLOAT = 5,        /**< 32-bit floating point */
  MINC2_DOUBLE = 6,       /**< 64-bit floating point */
  MINC2_STRING = 7,       /**< ASCII string (?) */
  MINC2_UBYTE = 100,      /**< 8-bit unsigned integer */
  MINC2_USHORT = 101,     /**< 16-bit unsigned integer */
  MINC2_UINT = 102,       /**< 32-bit unsigned integer */
  MINC2_SCOMPLEX = 1000,  /**< 16-bit signed integer complex */
  MINC2_ICOMPLEX = 1001,  /**< 32-bit signed integer complex */
  MINC2_FCOMPLEX = 1002,  /**< 32-bit floating point complex */
  MINC2_DCOMPLEX = 1003,  /**< 64-bit floating point complex */
  MINC2_MAX_TYPE_ID,
  MINC2_UNKNOWN  = -1     /**< when the type is a record */
};

/**
 * minc2 dimension information
 */
struct minc2_dimension
{
  int    id;             /**< dimension id, see enum minc2_dimensions*/
  int    length;         /**< dimension length */
  int    irregular;      /**< flag to show irregular sampling */
  double step;           /**< dimension step   */
  double start;          /**< dimension start  */
  int    have_dir_cos;   /**< flag that dimension cosines is valid*/
  double dir_cos[3];     /**< direction cosines*/
};


/**
 * minc2 error codes, compatible with minc2 API
 */
enum { MINC2_SUCCESS=0,MINC2_ERROR=-1};
/** Opaque structure representing minc2 file
 * 
 */
struct minc2_file;
typedef struct minc2_file* minc2_file_handle; 

/** 
 * allocate empty minc2 file structure, no need to call minc2_init after
 */
int minc2_allocate(minc2_file_handle * h);

minc2_file_handle minc2_allocate0(void);

int minc2_destroy(minc2_file_handle h);

/** 
 * initialize minc2 file structure
 */
int minc2_init(minc2_file_handle h);

/** 
 * deallocate minc2 file structure
 * will call standard free on it
 */
int minc2_free(minc2_file_handle h);

/**
 * open existing file
 */
int minc2_open(minc2_file_handle h,const char * path);

/**
 * define a new minc2 volume, using provided storage dimension information and storage data type
 */
int minc2_define(minc2_file_handle h, struct minc2_dimension *store_dims, int store_data_type,int data_type);

/**
 * create a new file, owerwriting an existing one if needed
 */
int minc2_create(minc2_file_handle h,const char * path);

/**
 * close file, flushing data to disk
 * sets minccomplete flag too
 */
int minc2_close(minc2_file_handle h);

/**
 * query number of dimensions
 */
int minc2_ndim(minc2_file_handle h,int *ndim);

/**
 * query total number of voxels
 */
int minc2_nelement(minc2_file_handle h,int *nelement);

/**
 * query data type, used to represent data
 */
int minc2_data_type(minc2_file_handle h,int *_type);

/**
 * query data type, used to store data on disk
 */
int minc2_storage_data_type(minc2_file_handle h,int *_type);

/**
 * query number of slice dimensions 
 */
int minc2_slice_ndim(minc2_file_handle h,int *slice_ndim);

/**
 * Setup minc file for reading or writing information
 * in standardized order ( Vector dimension - X - Y -Z -TIME )
 * with positive steps
 */
int minc2_setup_standard_order(minc2_file_handle h);

/**
 * get dimension information in current representation format
 */
int minc2_get_representation_dimensions(minc2_file_handle h,struct minc2_dimension **dims);

/**
 * get dimension information in file format
 */
int minc2_get_store_dimensions(minc2_file_handle h,struct minc2_dimension **dims);

/**
 * Load complete volume into memory
 */
int minc2_load_complete_volume(minc2_file_handle h,void *buffer,int representation_type);

/**
 * Save complete volume into memory
 */
int minc2_save_complete_volume(minc2_file_handle h,const void *buffer,int representation_type);

/**
 * Specify flags to use scaling
 * this have to be set before minc2_create
 */
int minc2_set_scaling(minc2_file_handle h,int use_global_scaling,int use_slice_scaling);

/**
 * Specify volume range, only when using hyperslab writing
 * Implies no slice scaling 
 */
int minc2_set_volume_range(minc2_file_handle h,double value_min,double value_max);

/**
 * Specify slice range, only when using hyperslab writing with slice scaling
 * Implies no slice scaling
 */
int minc2_set_slice_range(minc2_file_handle h,int *start,double value_min,double value_max);

/**
 * convert world X,Y,Z coordinates to voxel indexes (also in X,Y,Z order)
 */
int minc2_world_to_voxel(minc2_file_handle h,const double *world,double *voxel);

/**
 * convert voxel X,Y,Z indexes to world coordinates (also in X,Y,Z order)
 */
int minc2_voxel_to_world(minc2_file_handle h,const double *voxel,double *world);


/**
 * transfer attributes from one volume to another
 */
int minc2_copy_metadata(minc2_file_handle src,minc2_file_handle dst);

/**
 * write hyperslab, using current dimension order
 * WARNING: no range checks performed!
 */
int minc2_write_hyperslab(minc2_file_handle h,int *start,int *count,const void* buffer,int representation_type);

/**
 * read hyperslab, using current dimension order
 * WARNING: no range checks performed!
 */
int minc2_read_hyperslab(minc2_file_handle h,int *start,int *count,void* buffer,int representation_type);

/**
 * return human-readable type name
 */
const char * minc2_data_type_name(int minc2_type_id);

/**
 * return human-readable dimension name
 */
const char * minc2_dim_type_name(int minc2_dim_id);    
    """
    )


if __name__ == "__main__":
    ffi.compile(verbose=True)
    