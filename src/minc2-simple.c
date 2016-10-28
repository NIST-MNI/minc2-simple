#include "minc2.h"
#include "minc_config.h"

#include "minc2-simple.h"
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <limits.h>
/**
 * internal functions
 */
static int _minc2_allocate_dimensions(minc2_file_handle h,int nDims);
static int _minc2_cleanup_dimensions(minc2_file_handle h);
static mitype_t _minc2_type_to_mitype(int minc2_type);
static int _mitype_to_minc2_type(mitype_t t);

/**
 * internal representation of the minc file
 */
struct minc2_file {
  mihandle_t vol;

  int            ndims;
  int            store_type;  /*how data is stored in minc2 file*/
  int            data_type;   /*how data should be interpreted*/

  char         **dimension_name;
  misize_t      *dimension_size;
  double        *dimension_start;
  double        *dimension_step;

  struct         minc2_dimension *store_dims;
  struct         minc2_dimension *representation_dims;
 
  midimhandle_t *file_dims;
  midimhandle_t *apparent_dims;
  
  miboolean_t    slice_scaling_flag;
  miboolean_t    global_scaling_flag;
  
  miboolean_t    using_apparent_order;
  
  /*internal temporary data*/
  misize_t      *tmp_start;
  misize_t      *tmp_count;
};

/**
 * public functions
 */
int minc2_allocate(minc2_file_handle * h)
{
  *h=(struct minc2_file*)calloc(1,sizeof(struct minc2_file));
  return *h==NULL?MINC2_ERROR:MINC2_SUCCESS;
}

minc2_file_handle minc2_allocate0(void)
{
  minc2_file_handle h;
  if(minc2_allocate(&h)!=MINC2_SUCCESS)
    return NULL;
  return h;
}

int minc2_destroy(minc2_file_handle h)
{
  if(h->vol)
    minc2_close(h);
  return minc2_free(h);
}


int minc2_init(minc2_file_handle h)
{
  memset(h,0,sizeof(struct minc2_file));
  return MINC2_SUCCESS;
}

int minc2_free(minc2_file_handle h)
{
  if(!h)
    return MINC2_SUCCESS;
  _minc2_cleanup_dimensions(h);
  free(h);
  return MINC2_SUCCESS;
}

int minc2_open(minc2_file_handle h, const char * path)
{
  /*voxel valid range*/
  double valid_min,valid_max;
  
  /*real volume range, only awailable when slice scaling is off*/
  double volume_min=0.0,volume_max=1.0;
  miclass_t volume_data_class;
  mitype_t  store_type;
  int n_dims;
  int i;
  
  if ( miopen_volume(path, MI2_OPEN_READ, &h->vol) < 0 )
    return MINC2_ERROR;
  
  if ( miget_volume_dimension_count(h->vol, MI_DIMCLASS_ANY, MI_DIMATTR_ALL, &n_dims)<0)
    return MINC2_ERROR;
  
  if( _minc2_allocate_dimensions(h,n_dims)<0)
    return MINC2_ERROR;
  
  if ( miget_volume_dimensions(h->vol, MI_DIMCLASS_ANY, MI_DIMATTR_ALL, MI_DIMORDER_FILE, h->ndims,
                               h->file_dims) < 0 )
    return MINC2_ERROR;
  
  if ( miget_dimension_sizes( h->file_dims, h->ndims, h->dimension_size ) < 0 )
    return MINC2_ERROR;

  if ( miget_dimension_separations(h->file_dims, MI_ORDER_FILE, h->ndims, h->dimension_step) < 0 )
    return MINC2_ERROR;

  if ( miget_dimension_starts(h->file_dims, MI_ORDER_FILE, h->ndims, h->dimension_start) < 0 )
    return MINC2_ERROR;
  
  if ( miget_data_type(h->vol, &store_type) < 0 )
    return MINC2_ERROR;
  
  h->store_type=_mitype_to_minc2_type(store_type);
  
  if ( miget_slice_scaling_flag(h->vol, &h->slice_scaling_flag) < 0 )
    return MINC2_ERROR;

  if(miget_volume_valid_range(h->vol,&valid_max,&valid_min) < 0 )
    return MINC2_ERROR;

  if( !h->slice_scaling_flag )
  {
    if( miget_volume_range(h->vol,&volume_max,&volume_min) < 0 )
      return MINC2_ERROR;

    h->global_scaling_flag= !(volume_min == valid_min && volume_max == valid_max);
  }

  /*get dimension information*/
  for (i = 0; i < h->ndims; i++ )
  {
    char *      name;
    miboolean_t _sampling;
    
    /*const char *_sign="+";*/

    if ( miget_dimension_name(h->file_dims[i], &name) < 0 )
      return MINC2_ERROR;

    h->dimension_name[i] = name;
    
    h->store_dims[h->ndims-i-1].length=h->dimension_size[i];
    
    miget_dimension_separation(h->file_dims[i],MI_FILE_ORDER,&h->store_dims[h->ndims-i-1].step);
    
    if(miget_dimension_cosines(h->file_dims[i],&h->store_dims[h->ndims-i-1].dir_cos[0])==MI_NOERROR)
      h->store_dims[h->ndims-i-1].have_dir_cos=1;
    else
      h->store_dims[h->ndims-i-1].have_dir_cos=0;
    
    miget_dimension_start(h->file_dims[i],MI_FILE_ORDER,&h->store_dims[h->ndims-i-1].start);
    miget_dimension_sampling_flag(h->file_dims[i],&_sampling);
    
    h->store_dims[h->ndims-i-1].irregular=_sampling; /*documentation is wrong*/
    
    if(!strcmp(name,MIxspace) || !strcmp(name,MIxfrequency) ) /*this is X space*/
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_X;
    }
    else if(!strcmp(name,MIyspace) || !strcmp(name,MIyfrequency) ) /*this is Y
                                                                      space */
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_Y;
    }
    else if(!strcmp(name,MIzspace) || !strcmp(name,MIzfrequency) ) /*this is Z
                                                                    space*/
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_Z;
    }
    else if(!strcmp(name,MIvector_dimension) ) /*this is vector space*/
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_VEC;
    }
    else if(!strcmp(name,MItime) || !strcmp(name,MItfrequency) ) /*this is time
                                                                   space */
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_TIME;
    }
    else
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_TIME;
      MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported dimension type:%s",name);
      return MINC2_ERROR;
    }
  }
  /*mark the end of dimensions*/
  h->store_dims[h->ndims].id=MINC2_DIM_END;
  
  /*copy store to reprenetation dimension*/
  memmove(h->representation_dims,h->store_dims,sizeof(struct minc2_dimension)*(h->ndims+1));

  if ( miget_data_class(h->vol, &volume_data_class) < 0 )
    return MINC2_ERROR;  

  /* set the file data type*/
  if(h->slice_scaling_flag || h->global_scaling_flag)
  {
    switch ( h->store_type )
    {
      case MI_TYPE_FLOAT:
        h->data_type=MINC2_FLOAT;
        break;
      case MI_TYPE_DOUBLE:
        h->data_type=MINC2_DOUBLE;
        break;
      case MI_TYPE_FCOMPLEX:
        h->data_type=MINC2_FCOMPLEX;
        break;
      case MI_TYPE_DCOMPLEX:
        h->data_type=MINC2_DCOMPLEX;
        break;
      default:
        h->data_type=MINC2_FLOAT;
        break;
    } 
  }
  else /*not using normalization*/
  {
    switch ( h->store_type )
    {
      case MI_TYPE_BYTE:
        h->data_type=MINC2_BYTE;
        break;
      case MI_TYPE_UBYTE:
        h->data_type=MINC2_UBYTE;
        break;
      case MI_TYPE_SHORT:
        h->data_type=MINC2_SHORT;
        break;
      case MI_TYPE_USHORT:
        h->data_type=MINC2_USHORT;
        break;
      case MI_TYPE_INT:
        h->data_type=MINC2_INT;
        break;
      case MI_TYPE_UINT:
        h->data_type=MINC2_UINT;
        break;
      case MI_TYPE_FLOAT:
        h->data_type=MINC2_FLOAT;
        break;
      case MI_TYPE_DOUBLE:
        h->data_type=MINC2_DOUBLE;
        break;
      case MI_TYPE_SCOMPLEX:
        h->data_type=MINC2_SCOMPLEX;
        break;
      case MI_TYPE_ICOMPLEX:
        h->data_type=MINC2_ICOMPLEX;
        break;
      case MI_TYPE_FCOMPLEX:
        h->data_type=MINC2_FCOMPLEX;
        break;
      case MI_TYPE_DCOMPLEX:
        h->data_type=MINC2_DCOMPLEX;
        break;
      default:
        MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported file data type");
        return MINC2_ERROR;
    } 
  }

  switch ( volume_data_class )
  {
    case MI_CLASS_REAL:
    case MI_CLASS_INT:
    case MI_CLASS_LABEL: /* create an array of label names and values ?*/
/*      if(numberOfComponents == 1)
        {
        h->SetPixelType(SCALAR);
        }
      else
        {
        h->SetPixelType(VECTOR); 
        }*/
      break;
    case MI_CLASS_COMPLEX:
      /*h->SetPixelType(COMPLEX);*/
      /*numberOfComponents *= 2;*/
      break;
    default:
      MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported data class");
      return MINC2_ERROR;
  } //end of switch

  return MINC2_SUCCESS;
}


int minc2_slice_ndim(minc2_file_handle h,int *slice_ndim)
{
  if(h->slice_scaling_flag)
  {
    if( miget_slice_dimension_count(h->vol,MI_DIMCLASS_ANY, MI_DIMATTR_ALL, slice_ndim)<0)
      return MINC2_ERROR;
  } else {
    /*we don't have slice scaling?*/
    *slice_ndim= (h->ndims>2?2:h->ndims);
  }
  return MINC2_SUCCESS;
}


int minc2_setup_standard_order(minc2_file_handle h)
{
  /*int spatial_dimension=0;*/
  int usable_dimensions=0;
  int i;
  int dimension_indeces[5]={-1, -1, -1, -1, -1};
  
  if(!h->store_dims)
  {
    /*minc file is not opened or created yet*/
    return MINC2_ERROR;
  }
  
  /*create mapping*/
  for(i=0; i< h->ndims; i++)
  {
    switch(h->store_dims[i].id)
    {
      case MINC2_DIM_X:
        dimension_indeces[1]=i;
        break;
      case MINC2_DIM_Y:
        dimension_indeces[2]=i;
        break;
      case MINC2_DIM_Z:
        dimension_indeces[3]=i;
        break;
      case MINC2_DIM_VEC:
        dimension_indeces[0]=i;
        break;
      case MINC2_DIM_TIME:
        dimension_indeces[4]=i;
        break;
      default:
        /*error?*/
        MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported dimension");
        break;
    }
  }
  
  /*remap dimensions*/
  for(i=0; i<5; i++)
  {
    if( dimension_indeces[i]!=-1 )
    {
      h->apparent_dims[h->ndims-1-usable_dimensions]=h->file_dims[h->ndims-1-dimension_indeces[i]];
      
      /*always use positive, unless it is a vector dimension?*/
      if(i>0)
        miset_dimension_apparent_voxel_order(h->apparent_dims[h->ndims-1-usable_dimensions],MI_POSITIVE);
      
      h->representation_dims[usable_dimensions] = h->store_dims[dimension_indeces[i]];
      
      miget_dimension_separation(h->apparent_dims[h->ndims-1-usable_dimensions],MI_POSITIVE,&h->representation_dims[usable_dimensions].step);
      miget_dimension_start(     h->apparent_dims[h->ndims-1-usable_dimensions],MI_POSITIVE,&h->representation_dims[usable_dimensions].start);
      
      /*
      if(i>0 && i<4)
        spatial_dimension++;
      */
      usable_dimensions++;
    }
  }
  /*Set apparent dimension order to the MINC2 api*/
  if(miset_apparent_dimension_order(h->vol, usable_dimensions, h->apparent_dims)<0)
    return MINC2_ERROR;  
  h->using_apparent_order=1;
  return MINC2_SUCCESS;
}

int minc2_close(minc2_file_handle h)
{
  if(h->vol)
  {
    if ( miclose_volume(h->vol) < 0 )
      return MINC2_ERROR;
    
    h->vol=0;
    
    return _minc2_cleanup_dimensions(h);
  } else {
    /*File was not open?*/
    return _minc2_cleanup_dimensions(h);
  }
}

int minc2_ndim(minc2_file_handle h,int *ndim)
{
  if(h->ndims)
  {
    *ndim=h->ndims;
  } else 
    return MINC2_ERROR;
  return MINC2_SUCCESS;
}

int minc2_nelement(minc2_file_handle h,int *nelement)
{
  if(h->ndims && h->dimension_size)
  {
    int i;
    *nelement=1;
    for (i = 0; i < h->ndims; i++ )
    {
      *nelement *= h->dimension_size[i];
    }
    return MINC2_SUCCESS; 
  } else 
    return MINC2_ERROR;
}

int minc2_load_complete_volume( minc2_file_handle h,void *buffer,int representation_type)
{
  mitype_t buffer_type=MI_TYPE_UBYTE;
  int i;
  int err=MINC2_SUCCESS;

  if(h->using_apparent_order)
  {
    /*need to specify dimensions in apparent order, with minc2 convention that fasted dimensions are last*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      h->tmp_start[i]=0;
      h->tmp_count[i]=h->representation_dims[h->ndims-i-1].length;
    }
  } else {
    /*will read information on file order*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      h->tmp_start[i]=0;
      h->tmp_count[i]=h->store_dims[h->ndims-i-1].length;;
    }
  }
  buffer_type=_minc2_type_to_mitype(representation_type);

  if ( miget_real_value_hyperslab(h->vol, buffer_type, h->tmp_start, h->tmp_count, buffer) < 0 )
    err=MINC2_ERROR;
  else
    err=MINC2_SUCCESS;
  
  return err;
}

#define \
_GET_BUFFER_MIN_MAX(type_out,buffer,buffer_length,buffer_min,buffer_max) \
  { \
    size_t _i;\
    const type_out *_buffer = (const type_out *)buffer; \
    buffer_min=buffer_max=_buffer[0]; \
    for(_i=0;_i<buffer_length;_i++,_buffer++)\
    {\
      if( *_buffer > buffer_max ) buffer_max=*_buffer; \
      if( *_buffer < buffer_min ) buffer_min=*_buffer; \
    }\
  }


int minc2_save_complete_volume( minc2_file_handle h,const void *buffer,int representation_type)
{
  mitype_t buffer_type=MI_TYPE_UBYTE;
  /*mitype_t file_store_type=MI_TYPE_UBYTE;*/
  
  int i;
  int err=MINC2_SUCCESS;
  size_t   buffer_length=1;
  double   buffer_min,buffer_max;
  
  if(h->using_apparent_order)
  {
    /*need to specify dimensions in apparent order, with minc2 convention that fasted dimensions are last*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      h->tmp_start[i]=0;
      h->tmp_count[i]=h->representation_dims[h->ndims-i-1].length;
      buffer_length*=h->tmp_count[i];
    }
  } else {
    /*will write information in file order*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      h->tmp_start[i]=0;
      h->tmp_count[i]=h->store_dims[h->ndims-i-1].length;;
      buffer_length*=h->tmp_count[i];
    }
  }
  
  buffer_type=_minc2_type_to_mitype(representation_type);
  switch(representation_type )
  {
    case MINC2_UBYTE:
      _GET_BUFFER_MIN_MAX(unsigned char,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_BYTE:
      _GET_BUFFER_MIN_MAX(char,buffer, buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_USHORT:
      _GET_BUFFER_MIN_MAX(unsigned short,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_SHORT:
      _GET_BUFFER_MIN_MAX(short,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_UINT:
      _GET_BUFFER_MIN_MAX(unsigned int,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_INT:
      _GET_BUFFER_MIN_MAX(int,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_FLOAT:
      _GET_BUFFER_MIN_MAX(float,buffer,buffer_length,buffer_min,buffer_max);
      break;
    case MINC2_DOUBLE:
      _GET_BUFFER_MIN_MAX(double,buffer,buffer_length,buffer_min,buffer_max);
      break;
    default:
      MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported volume data type");
      return MINC2_ERROR;
  }
  
  if(minc2_set_volume_range(h,buffer_min,buffer_max)!=MINC2_SUCCESS)
    return MINC2_ERROR;

  if ( miset_real_value_hyperslab(h->vol, buffer_type, h->tmp_start, h->tmp_count, (void*)buffer ) < 0 )
    err=MINC2_ERROR;
  else
    err=MINC2_SUCCESS;
  
  return err;  
}

int minc2_set_scaling(minc2_file_handle h,int use_global_scaling,int use_slice_scaling)
{
  int err=MINC2_SUCCESS;

  if(use_global_scaling&&use_slice_scaling)
    /*can't have it both ways*/
    return MINC2_ERROR;

  h->global_scaling_flag=use_global_scaling;
  h->slice_scaling_flag=use_slice_scaling;

  return err;
}


int minc2_set_volume_range(minc2_file_handle h,
                           double value_min,
                           double value_max)
{
  int err=MINC2_SUCCESS;
  
  if( !h->global_scaling_flag )
  {
    
    if(miset_volume_valid_range( h->vol, value_max, value_min)<0) err=MINC2_ERROR;
    if(miset_volume_range(       h->vol, value_max, value_min)<0) err=MINC2_ERROR;
  }
  else // we are using scaling
  {
    if(miset_volume_range(h->vol,value_max,value_min)<0) err=MINC2_ERROR;
  }
  return err;
}

int minc2_set_slice_range(minc2_file_handle h,int *start,double value_min,double value_max)
{
  int i;
  for ( i = 0; i < h->ndims ; i++ )
  {
    h->tmp_start[i]=start[h->ndims-i-1];
  }
  if( miset_slice_range(h->vol,h->tmp_start, (size_t)h->ndims, value_min, value_max) < 0)
    return MINC2_ERROR;

  return MINC2_SUCCESS;
}


int minc2_write_hyperslab(minc2_file_handle h,int *start,int *count,const void* buffer,int representation_type)
{
  mitype_t buffer_type=_minc2_type_to_mitype(representation_type);
  int i;
  int err=MINC2_SUCCESS;
  
  /*need to specify dimensions with minc2 convention that fasted dimensions are last*/
  for ( i = 0; i < h->ndims ; i++ )
  {
    h->tmp_start[i]=start[h->ndims-i-1];
    h->tmp_count[i]=count[h->ndims-i-1];
  }

  if ( miset_real_value_hyperslab(h->vol, buffer_type, h->tmp_start, h->tmp_count, (void*)buffer ) < 0 )
    err=MINC2_ERROR;
  else
    err=MINC2_SUCCESS;

  return err;
}

int minc2_read_hyperslab(minc2_file_handle h,int *start,int *count,void* buffer,int representation_type)
{
  mitype_t buffer_type=_minc2_type_to_mitype(representation_type);
  int i;
  int err=MINC2_SUCCESS;

  /*need to specify dimensions with minc2 convention that fasted dimensions are last*/
  for ( i = 0; i < h->ndims ; i++ )
  {
    h->tmp_start[i]=start[h->ndims-i-1];
    h->tmp_count[i]=count[h->ndims-i-1];
  }

  if ( miget_real_value_hyperslab(h->vol, buffer_type, h->tmp_start, h->tmp_count, buffer) < 0 )
    err=MINC2_ERROR;
  else
    err=MINC2_SUCCESS;

  return err;
}

int minc2_data_type(minc2_file_handle h,int *_type)
{
  if(h->data_type>0)
  {
    *_type=h->data_type;
    return MINC2_SUCCESS;
  } else {
    /*not initialized!*/
    return MINC2_ERROR;
  }
}

int minc2_storage_data_type(minc2_file_handle h,int *_type)
{
  if(h->data_type>0)
  {
    *_type=(int)h->store_type;
    return MINC2_SUCCESS;
  } else {
    /*not initialized!*/
    return MINC2_ERROR;
  }
}

int minc2_get_representation_dimensions(minc2_file_handle h,struct minc2_dimension **dims)
{
  if(!h->representation_dims)
    return MINC2_ERROR;
  *dims=h->representation_dims;
  return MINC2_SUCCESS;
}

int minc2_get_store_dimensions(minc2_file_handle h,struct minc2_dimension **dims)
{
  if(!h->store_dims)
    return MINC2_ERROR;
  *dims=h->store_dims;
  return MINC2_SUCCESS;
}


int minc2_define(minc2_file_handle h, struct minc2_dimension *store_dims, int store_data_type,int data_type)
{
  int i;
  int ndims=0;
  struct minc2_dimension * dim;
  /*figure out number of dimension*/
  for(dim=store_dims;dim->id!=MINC2_DIM_END;dim++)
  {
    ndims++;
  }
  
  h->store_type=store_data_type; /*TODO: add mapping?*/
  h->data_type=data_type;
  
  /*TODO: add more cases*/
  if( ( h->store_type==MI_TYPE_FLOAT ||
        h->store_type==MI_TYPE_DOUBLE ) || 
        
      ( 
        ( h->store_type==MI_TYPE_BYTE  || h->store_type==MI_TYPE_INT  || h->store_type==MI_TYPE_SHORT ||
          h->store_type==MI_TYPE_UBYTE || h->store_type==MI_TYPE_UINT || h->store_type==MI_TYPE_USHORT ) && 
        ( h->data_type==MI_TYPE_BYTE   || h->data_type==MI_TYPE_INT   || h->data_type==MI_TYPE_SHORT ||
          h->data_type==MI_TYPE_UBYTE  || h->data_type==MI_TYPE_UINT  || h->data_type==MI_TYPE_USHORT )
      )
    )
  {
    h->slice_scaling_flag=0;
    h->global_scaling_flag=0;
  } else {
    h->slice_scaling_flag=0; /*TODO: use slice scaling sometimes?*/
    h->global_scaling_flag=1;
  }
  
  _minc2_allocate_dimensions(h,ndims);
  memmove(h->store_dims         ,store_dims,sizeof(struct minc2_dimension)*(h->ndims+1));
  memmove(h->representation_dims,store_dims,sizeof(struct minc2_dimension)*(h->ndims+1));
  
  for(dim=store_dims,i=0;dim->id!=MINC2_DIM_END;dim++,i++)
  {
    switch(dim->id)
    {
    case MINC2_DIM_X:
      micreate_dimension(MIxspace,MI_DIMCLASS_SPATIAL, 
                         dim->irregular?MI_DIMATTR_NOT_REGULARLY_SAMPLED:MI_DIMATTR_REGULARLY_SAMPLED, 
                         dim->length,
                         &h->file_dims[ndims-i-1] );
      break;
    case MINC2_DIM_Y:
      micreate_dimension(MIyspace,MI_DIMCLASS_SPATIAL, 
                         dim->irregular?MI_DIMATTR_NOT_REGULARLY_SAMPLED:MI_DIMATTR_REGULARLY_SAMPLED, 
                         dim->length,
                         &h->file_dims[ndims-i-1] );
      break;
    case MINC2_DIM_Z:
      micreate_dimension(MIzspace,MI_DIMCLASS_SPATIAL, 
                         dim->irregular?MI_DIMATTR_NOT_REGULARLY_SAMPLED:MI_DIMATTR_REGULARLY_SAMPLED, 
                         dim->length,
                         &h->file_dims[ndims-i-1] );
      break;
    case MINC2_DIM_TIME:
      micreate_dimension(MItime, MI_DIMCLASS_TIME, 
                         dim->irregular?MI_DIMATTR_NOT_REGULARLY_SAMPLED:MI_DIMATTR_REGULARLY_SAMPLED, 
                         dim->length,
                         &h->file_dims[ndims-i-1] );
      break;
    case MINC2_DIM_VEC:
      micreate_dimension(MIvector_dimension,MI_DIMCLASS_RECORD, MI_DIMATTR_REGULARLY_SAMPLED, 
                         dim->length,
                         &h->file_dims[ndims-i-1] );
      break;
    default:
      /*don't know this dimension type*/
      /*TODO: report error*/
      break;
    }
    miset_dimension_start(     h->file_dims[ndims-i-1],dim->start);
    miset_dimension_separation(h->file_dims[ndims-i-1],dim->step );
    if(dim->have_dir_cos)
      miset_dimension_cosines( h->file_dims[ndims-i-1],dim->dir_cos);
  }
  return MINC2_SUCCESS;
}


int minc2_create(minc2_file_handle h,const char * path)
{
  int err;
  /**/
  mivolumeprops_t hprops;
  
  if( minew_volume_props(&hprops) < 0)
  {
    return MINC2_ERROR;
  }
  
  /*TODO: move it to volume definition*/
  if(miget_cfg_present(MICFG_COMPRESS) && miget_cfg_int(MICFG_COMPRESS)>0  )
  {
    if(miset_props_compression_type(hprops, MI_COMPRESS_ZLIB)<0)
    {
      return MINC2_ERROR;
    }

    if(miset_props_zlib_compression(hprops,miget_cfg_int(MICFG_COMPRESS))<0)
    {
      return MINC2_ERROR;
    }
  }
  else
  {
    if(miset_props_compression_type(hprops, MI_COMPRESS_NONE)<0)
    {
      return MINC2_ERROR;
    }
  }


  if ( micreate_volume ( path, h->ndims, h->file_dims, h->store_type,
                         MI_CLASS_REAL, hprops, &h->vol )<0 ) /*change MI_CLASS_REAL to something else?*/
  {
    MI_LOG_ERROR(MI2_MSG_GENERIC,"Couldn't open file %s",path);
    return MINC2_ERROR;
  }

  /*have to set slice scaling flag before image is allocated*/
  if ( miset_slice_scaling_flag(h->vol, h->slice_scaling_flag )<0 )
  {
    MI_LOG_ERROR(MI2_MSG_GENERIC,"Couldn't set slice scaling");
    return MINC2_ERROR;
  }

  if (  micreate_volume_image ( h->vol ) <0 )
  {
    MI_LOG_ERROR(MI2_MSG_GENERIC,"Couldn't create image in file %s",path);
    return MINC2_ERROR;
  }

  if(h->global_scaling_flag)
  {
    switch(h->store_type)
    {
      case MI_TYPE_BYTE:
        if(miset_volume_valid_range(h->vol,SCHAR_MAX,SCHAR_MIN)<0) err=MINC2_ERROR;
        break;
      case MI_TYPE_UBYTE:
        if(miset_volume_valid_range(h->vol,UCHAR_MAX,0)<0) err=MINC2_ERROR;
        break;
      case MI_TYPE_SHORT:
        if(miset_volume_valid_range(h->vol,SHRT_MAX,SHRT_MIN)<0) err=MINC2_ERROR;
        break;
      case MI_TYPE_USHORT:
        if(miset_volume_valid_range(h->vol,USHRT_MAX,0)<0) err=MINC2_ERROR;
        break;
      case MI_TYPE_INT:
        if(miset_volume_valid_range(h->vol,INT_MAX,INT_MIN)<0) err=MINC2_ERROR;
        break;
      case MI_TYPE_UINT:
        if(miset_volume_valid_range(h->vol,UINT_MAX,0)<0) err=MINC2_ERROR;
        break;
      default:
        /*error*/
        MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported store data type");
        return MINC2_ERROR;
    }
  }

  return err;
}


int minc2_world_to_voxel(minc2_file_handle h,const double *world,double *voxel)
{
  if(!h->vol)
    return MINC2_ERROR;

  if(miconvert_world_to_voxel(h->vol,world,voxel)<0)
    return MINC2_ERROR;

  return MINC2_SUCCESS;
}


int minc2_voxel_to_world(minc2_file_handle h,const double *voxel,double *world)
{
  if(!h->vol)
    return MINC2_ERROR;

  if(miconvert_voxel_to_world(h->vol,voxel,world)<0)
    return MINC2_ERROR;

  return MINC2_SUCCESS;
}


static int _minc2_cleanup_dimensions(minc2_file_handle h)
{
  int i;
  if( h->dimension_name )
  {
    for ( i = 0; i < h->ndims; i++ )
    {
      if(h->dimension_name[i]) 
        mifree_name( h->dimension_name[i] );
      h->dimension_name[i]=NULL;
    }
    free(h->dimension_name);
  }

  
  if(h->dimension_size)  free(h->dimension_size);
  if(h->dimension_start) free(h->dimension_start);
  if(h->dimension_step)  free(h->dimension_step);
  if(h->file_dims)       free(h->file_dims);
  if(h->apparent_dims)   free(h->apparent_dims);
  if(h->store_dims)      free(h->store_dims);
  if(h->representation_dims)free(h->representation_dims);
  if(h->tmp_start)       free(h->tmp_start);
  if(h->tmp_count)       free(h->tmp_count);
  
  h->dimension_name    = NULL;
  h->dimension_size    = NULL;
  h->dimension_start   = NULL;
  h->dimension_step    = NULL;
  h->file_dims         = NULL;
  h->apparent_dims     = NULL;
  h->store_dims        = NULL;
  h->representation_dims=NULL;
  h->using_apparent_order=0;
  h->tmp_count         =NULL;
  h->tmp_start         =NULL;
  
  return MINC2_SUCCESS;
}

static int _minc2_allocate_dimensions(minc2_file_handle h,int nDims)
{
  _minc2_cleanup_dimensions(h);

  h->ndims=nDims;

  h->dimension_name  = (char**)         calloc(h->ndims,sizeof(char*));
  h->dimension_size  = (misize_t*)      calloc(h->ndims,sizeof(misize_t));
  h->dimension_start = (double*)        calloc(h->ndims,sizeof(double));
  h->dimension_step  = (double*)        calloc(h->ndims,sizeof(double));
  h->file_dims       = (midimhandle_t*) calloc(h->ndims,sizeof(midimhandle_t));
  h->apparent_dims   = (midimhandle_t*) calloc(h->ndims,sizeof(midimhandle_t));
  h->store_dims      = (struct minc2_dimension*)calloc(h->ndims+1,sizeof(struct minc2_dimension));
  h->representation_dims= (struct minc2_dimension*)calloc(h->ndims+1,sizeof(struct minc2_dimension));

  h->tmp_start       = (misize_t *)calloc(h->ndims,sizeof(misize_t));
  h->tmp_count       = (misize_t *)calloc(h->ndims,sizeof(misize_t));

  /*TODO: check if memory was allocated?*/
  return MINC2_SUCCESS;
}

int minc2_copy_metadata(minc2_file_handle src,minc2_file_handle dst)
{
  milisthandle_t grplist;
  int err=0;
  if( !src->vol || !dst->vol)
    return MINC2_ERROR;

  if ( (milist_start(src->vol, "", 0, &grplist) ) == MI_NOERROR )
  {
      char           group_name[256];
      /*milisthandle_t attlist;*/
      while( milist_grp_next(grplist, group_name, sizeof(group_name) ) == MI_NOERROR )
      {
        if(micopy_attr(src->vol,group_name,dst->vol)<0)
          err++;
      }
    milist_finish(grplist);
  } else {
    return MINC2_ERROR;
  }
  /*TODO: copy history attribute, because micopy_attr doesn't copy it*/


  return err>0?MINC2_ERROR:MINC2_SUCCESS;
}


const char * minc2_data_type_name(int minc2_type_id)
{
  switch(minc2_type_id )
    {
    case MINC2_UBYTE:
      return "unsigned char";
    case MINC2_BYTE:
      return "char";
    case MINC2_USHORT:
      return "unsigned short";
    case MINC2_SHORT:
      return "short";
    case MINC2_UINT:
      return "unsigned int";
    case MINC2_INT:
      return "int";
    case MINC2_FLOAT:
      return "float";
    case MINC2_DOUBLE:
      return "double";
    default:
      return "Unknown";
  }
}

const char * minc2_dim_type_name(int minc2_dim_id)
{
  switch(minc2_dim_id)
  {
    case MINC2_DIM_X:
      return "X";
    case MINC2_DIM_Y:
      return "Y";
    case MINC2_DIM_Z:
      return "Z";
    case MINC2_DIM_TIME:
      return "Time";
    case MINC2_DIM_VEC:
      return "Vector";
    case MINC2_DIM_UNKNOWN:
    case MINC2_DIM_END:
    default:
      return "Unknown";
  }
}

static mitype_t _minc2_type_to_mitype(int minc2_type)
{
  /*this is identity transform at the moment*/
  return (mitype_t)minc2_type;
  /*
    switch(representation_type )
  {
    case MINC2_UBYTE:
      buffer_type=MI_TYPE_UBYTE;
      break;
    case MINC2_BYTE:
      buffer_type=MI_TYPE_BYTE;
      break;
    case MINC2_USHORT:
      buffer_type=MI_TYPE_USHORT;
      break;
    case MINC2_SHORT:
      buffer_type=MI_TYPE_SHORT;
      break;
    case MINC2_UINT:
      buffer_type=MI_TYPE_UINT;
      break;
    case MINC2_INT:
      buffer_type=MI_TYPE_INT;
      break;
    case MINC2_FLOAT:
      buffer_type=MI_TYPE_FLOAT;
      break;
    case MINC2_DOUBLE:
      buffer_type=MI_TYPE_DOUBLE;
      break;
    default:
      MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported volume data type");
      free( start ); 
      free( count );
      return MINC2_ERROR;
  }
   */
}

static int _mitype_to_minc2_type(mitype_t t)
{
  /*this is identity transform at the moment*/
  return (int)t;
}



/* kate: indent-mode cstyle; indent-width 2; replace-tabs on; remove-trailing-space on; hl c */