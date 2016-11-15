-- lua module to read and write minc2 files
-- using minc2-simple c glue
-- using FFI 
local ffi = require("ffi")
require('torch')

-- contents of ../src/minc2-simple.h :
ffi.cdef[[

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
 *
 */
struct minc2_info_iterator;
typedef struct minc2_info_iterator* minc2_info_iterator_handle;


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

/**
 * alternative version
 */
minc2_file_handle minc2_allocate0(void);


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
 * close minc2 file if it's open,
 * then deallocate minc2 file structure
 */
int minc2_destroy(minc2_file_handle h);


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


/**
 * get attribute type
 */
int minc2_get_attribute_type(minc2_file_handle h,const char* group,const char* attr,int *minc2_type);

/**
 * get attribute length
 */
int minc2_get_attribute_length(minc2_file_handle h,const char* group,const char* attr,int *attr_length);

/**
 * read attribute
 */
int minc2_read_attribute(minc2_file_handle h,const char* group,const char* attr,void *buf,int buf_size);

/**
 * write attribute
 */
int minc2_write_attribute(minc2_file_handle h,const char* group,const char* attr,const void *buf,int buf_size,int minc2_type);

/**
 * delete attribute
 */
int minc2_delete_attribute(minc2_file_handle h,const char* group,const char* attr);

/**
 * delete the whole group
 */
int minc2_delete_group(minc2_file_handle h,const char* group);

/**
 * Ititialize info iterator
 */
minc2_info_iterator_handle minc2_allocate_info_iterator(void);

/**
 * Free info iterator: stop iterator if needed and deallocate memory
 */
int minc2_free_info_iterator(minc2_info_iterator_handle it);

/**
 * Stop iterator: stop itarating , the iterator handle can be re-used for another time
 */
int minc2_stop_info_iterator(minc2_info_iterator_handle it);


/**
 * Start iterating over groups
 */
int minc2_start_group_iterator(minc2_file_handle h,minc2_info_iterator_handle group_it);

/**
 * Start iterating over attrobutes
 */
int minc2_start_attribute_iterator(minc2_file_handle h,const char* group,minc2_info_iterator_handle it);

/**
 * advance to next available group item
 */
int minc2_iterator_group_next(minc2_info_iterator_handle it);

/**
 * advance to next available attrobute item
 */
int minc2_iterator_attribute_next(minc2_info_iterator_handle it);

/**
 * Get current iterator contents
 */
const char* minc2_iterator_group_name(minc2_info_iterator_handle it);

/**
 * Get current iterator contents
 */
const char* minc2_iterator_attribute_name(minc2_info_iterator_handle it);

/**
 * generate timestamp
 */
char* minc2_timestamp(int argc,char **argv);

    ]]

local lib = ffi.load("minc2-simple") -- for now fixed path

minc2_file = {
    -- minc2 constants
    
    -- minc2 dimensions
    MINC2_DIM_UNKNOWN=ffi.C.MINC2_DIM_UNKNOWN,
    MINC2_DIM_X    = ffi.C.MINC2_DIM_X,
    MINC2_DIM_Y    = ffi.C.MINC2_DIM_Y,
    MINC2_DIM_Z    = ffi.C.MINC2_DIM_Z,
    MINC2_DIM_TIME = ffi.C.MINC2_DIM_TIME,
    MINC2_DIM_VEC  = ffi.C.MINC2_DIM_VEC,
    MINC2_DIM_END  = ffi.C.MINC2_DIM_END,
    
    -- minc2 data types
    MINC2_BYTE     = ffi.C.MINC2_BYTE ,
    MINC2_SHORT    = ffi.C.MINC2_SHORT ,
    MINC2_INT      = ffi.C.MINC2_INT ,
    MINC2_FLOAT    = ffi.C.MINC2_FLOAT ,
    MINC2_DOUBLE   = ffi.C.MINC2_DOUBLE ,
    MINC2_STRING   = ffi.C.MINC2_STRING ,
    MINC2_UBYTE    = ffi.C.MINC2_UBYTE ,
    MINC2_USHORT   = ffi.C.MINC2_USHORT ,
    MINC2_UINT     = ffi.C.MINC2_UINT ,
    MINC2_SCOMPLEX = ffi.C.MINC2_SCOMPLEX ,
    MINC2_ICOMPLEX = ffi.C.MINC2_ICOMPLEX ,
    MINC2_FCOMPLEX = ffi.C.MINC2_FCOMPLEX ,
    MINC2_DCOMPLEX = ffi.C.MINC2_DCOMPLEX ,
    MINC2_MAX_TYPE_ID=ffi.C.MINC2_MAX_TYPE_ID,
    MINC2_UNKNOWN  = ffi.C.MINC2_UNKNOWN  ,

    -- minc2 status
    MINC2_SUCCESS  = ffi.C.MINC2_SUCCESS,
    MINC2_ERROR    = ffi.C.MINC2_ERROR
}
minc2_file.__index = minc2_file

function minc2_file.new(path)
  local self = setmetatable({}, minc2_file)
  self._v=ffi.gc(lib.minc2_allocate0(),lib.minc2_destroy)
  if path~=nil then
      self:open(path)
  end
  return self
end

-- open existing minc2 file
function minc2_file:open(path)
    --print("Going to open:"..path)
    assert(path~=nil,"Provide minc2 file")
    assert( lib.minc2_open(self._v,path)==ffi.C.MINC2_SUCCESS )
end

-- close a minc2 file
function minc2_file:close()
    assert(lib.minc2_close(self._v)==ffi.C.MINC2_SUCCESS)
end

-- query number of dimensions
function minc2_file:ndim()
    dd=ffi.new("int[1]")
    assert(lib.minc2_ndim(self._v,dd)==ffi.C.MINC2_SUCCESS)
    return dd[0]
end

-- provide descriptor of dimensions
function minc2_file:store_dims()
    local dims=ffi.new("struct minc2_dimension*[1]")
    assert(lib.minc2_get_store_dimensions(self._v,dims)==ffi.C.MINC2_SUCCESS)
    return dims[0]
end

-- provide descriptor of dimensions
function minc2_file:representation_dims()
    local dims=ffi.new("struct minc2_dimension*[1]")
    assert(lib.minc2_get_representation_dimensions(self._v,dims)==ffi.C.MINC2_SUCCESS)
    return dims[0]
end


-- define a new volume
function minc2_file:define(dims,store_type,representation_type)
    assert(dims~=nil,"dims need to be defined")
    assert(store_type~=nil,"Store data type need to be set")
    assert(representation_type~=nil,"Data type need to be set")
    
    if type(dims)== "table" then
        --assume user didn't provide m2.MINC2_DIM_END
        local mydims={}
        for k, v in pairs(dims) do mydims[k] = v end
        mydims[#mydims]={id=minc2_file.MINC2_DIM_END }
        
        dims=ffi.new("struct minc2_dimension[?]",#mydims,mydims)
    end
    
    assert(lib.minc2_define(self._v,dims,store_type,representation_type)==ffi.C.MINC2_SUCCESS)
end

-- create a  new minc2 file
function minc2_file:create(path)
    assert( path~=nil )
    assert( lib.minc2_create(self._v, path ) == ffi.C.MINC2_SUCCESS)
end

function minc2_file:copy_metadata(another)
    assert(another)
    assert(lib.minc2_copy_metadata(another._v,self._v)==ffi.C.MINC2_SUCCESS)
end

-- function minc2_file:load_complete_volume(data_type)
--     data_type=data_type or ffi.C.MINC2_FLOAT
--     buf_len=ffi.new("int[1]")
--     lib.minc2_nelement(self._v,buf_len)
--     buf_len=buf_len[0]
--     buf=nil
--     if data_type==ffi.C.MINC2_BYTE then 
--         buf=ffi.new("int8_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_UBYTE then 
--         buf=ffi.new("uint8_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_SHORT then 
--         buf=ffi.new("int16_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_USHORT then 
--         buf=ffi.new("uint16_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_INT then 
--         buf=ffi.new("int32_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_UINT then 
--         buf=ffi.new("uint32_t[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_FLOAT then 
--         buf=ffi.new("float[?]",buf_len)
--     elseif data_type==ffi.C.MINC2_DOUBLE then 
--         buf=ffi.new("double[?]",buf_len)
--     else
--         assert(false,"Unsupported  yet")
--     end
--     assert(lib.minc2_load_complete_volume(self._v,buf,data_type)==ffi.C.MINC2_SUCCESS)
--     return buf
-- end

function minc2_file:load_complete_volume(data_type)
    -- will be torch tensors
    -- require('torch')
    data_type=data_type or ffi.C.MINC2_FLOAT
    -- local buf_len=ffi.new("int[1]")
    -- lib.minc2_nelement(self._v,buf_len)
    -- buf_len=buf_len[0]
    local buf=nil
    local _dims=self:representation_dims()
    local dims=torch.LongStorage(self:ndim())
    -- local nelements=1
    
    -- Torch tensor defines dimensions in a slowest first fashion
    for i=0,(self:ndim()-1) do 
        dims[self:ndim()-i]=_dims[i].length
        --nelements=nelements*_dims[i].length
    end
    
    if data_type==ffi.C.MINC2_BYTE then 
        buf=torch.CharTensor(dims)
    elseif data_type==ffi.C.MINC2_UBYTE then 
        buf=torch.ByteTensor(dims)
    elseif data_type==ffi.C.MINC2_SHORT then 
        buf=torch.ShortTensor(dims)
    elseif data_type==ffi.C.MINC2_USHORT then 
        buf=torch.ShortTensor(dims)
    elseif data_type==ffi.C.MINC2_INT then 
        buf=torch.IntTensor(dims)
    elseif data_type==ffi.C.MINC2_UINT then 
        buf=torch.IntTensor(dims)
    elseif data_type==ffi.C.MINC2_FLOAT then 
        buf=torch.FloatTensor(dims)
    elseif data_type==ffi.C.MINC2_DOUBLE then 
        buf=torch.DoubleTensor(dims)
    else
        error("Unsupported  yet")
    end
    assert( 
        lib.minc2_load_complete_volume(self._v, buf:storage():data(), data_type)==ffi.C.MINC2_SUCCESS 
    )
    
    return buf
end

function minc2_file:setup_standard_order()
    assert( lib.minc2_setup_standard_order(self._v) == ffi.C.MINC2_SUCCESS)
end


-- function minc2_file:save_complete_volume(buf,data_type)
--     data_type=data_type or ffi.C.MINC2_FLOAT
--     assert(buf~=nil)
--     assert(lib.minc2_save_complete_volume(self._v,buf,data_type)==ffi.C.MINC2_SUCCESS)
--     return buf
-- end


function minc2_file:save_complete_volume(buf)
    assert(buf~=nil)
    --local t=require('torch')
    local data_type=ffi.C.MINC2_FLOAT
    -- local s=buf:storage()
    local store_type=torch.type(buf)
    
    -- TODO: implement dimension checking!
    -- TODO: check if tensor is contigious
    -- TODO: figure out how to save non-contigious tensor
    
    if store_type == 'torch.CharTensor' then
        data_type=ffi.C.MINC2_BYTE
    elseif store_type=='torch.ByteTensor' then 
        data_type=ffi.C.MINC2_UBYTE
    elseif store_type=='torch.ShortTensor' then 
        data_type=ffi.C.MINC2_SHORT
    elseif store_type=='torch.ShortTensor' then 
        data_type=ffi.C.MINC2_USHORT
    elseif store_type=='torch.IntTensor' then 
        data_type=ffi.C.MINC2_INT
    elseif store_type == 'torch.IntTensor' then 
        data_type=ffi.C.MINC2_UINT
    elseif store_type == 'torch.FloatTensor' then 
        data_type=ffi.C.MINC2_FLOAT
    elseif store_type == 'torch.DoubleTensor' then 
        data_type=ffi.C.MINC2_DOUBLE
    else
        print(string.format("store_type=%s",store_type))
        error("Unsupported  yet")
    end
    
    assert(
        lib.minc2_save_complete_volume(self._v,buf:storage():data(),data_type)==ffi.C.MINC2_SUCCESS
        )
    return buf
end

function minc2_file:read_attribute(group,attribute)
    
    local attr_type=ffi.new("int[1]")
    local attr_length=ffi.new("int[1]")
    
    -- assume that if we can't get attribute type, it's missing, return nil
    
    if lib.minc2_get_attribute_type(self._v,group,attribute,attr_type)~=ffi.C.MINC2_SUCCESS then
        return nil
    end
    
    assert(lib.minc2_get_attribute_length(self._v,group,attribute,attr_length)==ffi.C.MINC2_SUCCESS)
    
    if attr_type[0] == ffi.C.MINC2_STRING then
        print("reading string attribute...")
        local buf = ffi.new("uint8_t[?]", attr_length[0])
        assert(lib.minc2_read_attribute(self._v,group,attribute,buf,attr_length[0])==ffi.C.MINC2_SUCCESS);
        return ffi.string(buf, attr_length[0])
    else
        local buf
        local dims=attr_length[0]
        local data_type=attr_type[0]
        
        if data_type==ffi.C.MINC2_BYTE then 
            buf=torch.CharTensor(dims)
        elseif data_type==ffi.C.MINC2_UBYTE then 
            buf=torch.ByteTensor(dims)
        elseif data_type==ffi.C.MINC2_SHORT then 
            buf=torch.ShortTensor(dims)
        elseif data_type==ffi.C.MINC2_USHORT then 
            buf=torch.ShortTensor(dims)
        elseif data_type==ffi.C.MINC2_INT then 
            buf=torch.IntTensor(dims)
        elseif data_type==ffi.C.MINC2_UINT then 
            buf=torch.IntTensor(dims)
        elseif data_type==ffi.C.MINC2_FLOAT then 
            buf=torch.FloatTensor(dims)
        elseif data_type==ffi.C.MINC2_DOUBLE then 
            buf=torch.DoubleTensor(dims)
        else
            error("Unsupported  yet:"..data_type)
        end

        assert(lib.minc2_read_attribute(self._v,group,attribute,buf:storage():data(),attr_length[0])==ffi.C.MINC2_SUCCESS);
        
        return buf
    end
end

function minc2_file:write_attribute(group,attribute,value)
    -- local attr_type=ffi.new("int[1]")
    -- local attr_length=ffi.new("int[1]")
    local dtype=type(value)
    
    if dtype=="string" then
        attr_type=ffi.C.MINC2_STRING
        attr_length=#value+1
        assert(
            lib.minc2_write_attribute(self._v,group,attribute,ffi.cast("const char[]",value),#value+1,ffi.C.MINC2_STRING)==ffi.C.MINC2_SUCCESS
            )
    else
        local _value=value
        if dtype=="table" then 
            _value=torch.Tensor(value)
        elseif dtype=="number" then
            _value=torch.Tensor(1)
            _value[1]=value
        end
        
        local store_type=torch.type(_value)
        local data_type
        
        if store_type == 'torch.CharTensor' then
            data_type=ffi.C.MINC2_BYTE
        elseif store_type=='torch.ByteTensor' then 
            data_type=ffi.C.MINC2_UBYTE
        elseif store_type=='torch.ShortTensor' then 
            data_type=ffi.C.MINC2_SHORT
        elseif store_type=='torch.ShortTensor' then 
            data_type=ffi.C.MINC2_USHORT
        elseif store_type=='torch.IntTensor' then 
            data_type=ffi.C.MINC2_INT
        elseif store_type == 'torch.IntTensor' then 
            data_type=ffi.C.MINC2_UINT
        elseif store_type == 'torch.FloatTensor' then 
            data_type=ffi.C.MINC2_FLOAT
        elseif store_type == 'torch.DoubleTensor' then 
            data_type=ffi.C.MINC2_DOUBLE
        else
            error("Unsupported  yet:"..store_type)
        end
        
        assert(
            lib.minc2_write_attribute(self._v,group,attribute,_value:storage():data(),_value:storage():size(),data_type)==ffi.C.MINC2_SUCCESS
            )
    end
end

function minc2_file:metadata()
    local ret={}
    
    local group_iterator=ffi.gc(lib.minc2_allocate_info_iterator(), minc2_free_info_iterator)
    local attr_iterator=ffi.gc(lib.minc2_allocate_info_iterator(),minc2_free_info_iterator)

    assert(lib.minc2_start_group_iterator(self._v,group_iterator)==ffi.C.MINC2_SUCCESS)
    
    while lib.minc2_iterator_group_next(group_iterator)==ffi.C.MINC2_SUCCESS do
        local gname=lib.minc2_iterator_group_name(group_iterator)
        assert(lib.minc2_start_attribute_iterator(self._v, gname, attr_iterator)==ffi.C.MINC2_SUCCESS)
        local g={}
        
        while lib.minc2_iterator_attribute_next(attr_iterator)==ffi.C.MINC2_SUCCESS do
            local aname=lib.minc2_iterator_attribute_name(attr_iterator)
            g[ ffi.string(aname) ] = self:read_attribute(gname, aname)
        end
        
        ret[ ffi.string(lib.minc2_iterator_group_name(group_iterator)) ] = g
        lib.minc2_stop_info_iterator(attr_iterator)
    end
    lib.minc2_stop_info_iterator(group_iterator)
    return ret
end


function minc2_file:write_metadata(m)
    
    local group_iterator=ffi.gc(lib.minc2_allocate_info_iterator(), minc2_free_info_iterator)
    local attr_iterator=ffi.gc(lib.minc2_allocate_info_iterator(),minc2_free_info_iterator)

    assert(lib.minc2_start_group_iterator(self._v,group_iterator)==ffi.C.MINC2_SUCCESS)
    local group,g
    for group,g in pairs(m) do
        local attr,a
        for attr,a in pairs(g) do
            self:write_attribute(group,attr,a)
        end
    end
end


-- this is all we have in the module
return { 
    -- minc2 file reader/writer
    minc2_file=minc2_file 
}
