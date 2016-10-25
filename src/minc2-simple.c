#include "minc2.h"

#include "minc2-simple.h"
#include <stdlib.h>
#include <string.h>
#include <string.h>

/**
 * internal functions
 */
static int _minc2_allocate_dimensions(minc2_file_handle h,int nDims);
static int _minc2_cleanup_dimensions(minc2_file_handle h);

/**
 * internal representation of the minc file
 */
struct minc2_file {
  mihandle_t vol;
  
  int            ndims;
  mitype_t       store_type;/*how data is stored in minc2 file*/
  int            data_type; /*how data should be interpreted*/
  
  char         **dimension_name;
  misize_t      *dimension_size;
  double        *dimension_start;
  double        *dimension_step;
  int            dimension_indeces[5];
  
  struct         minc2_dimension *store_dims;
  struct         minc2_dimension *representation_dims;
 
  midimhandle_t *file_dims;
  midimhandle_t *apparent_dims;
  
  miboolean_t    slice_scaling_flag;
  miboolean_t    global_scaling_flag;
  
  miboolean_t    using_apparent_order;
};

/**
 * public functions
 */
int minc2_allocate(minc2_file_handle * h)
{
  *h=(struct minc2_file*)calloc(1,sizeof(struct minc2_file));
  return *h==NULL?MINC2_ERROR:MINC2_SUCCESS;
}

int minc2_init(minc2_file_handle h)
{
  memset(h,0,sizeof(struct minc2_file));
  return MINC2_SUCCESS;
}

int minc2_free(minc2_file_handle h)
{
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
  int spatial_dimension_count=0;
  int spatial_dimension=0;
  int usable_dimensions=0;
  miclass_t volume_data_class;

  int i;
  int n_dims;
  
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
  
  if ( miget_data_type(h->vol, &h->store_type) < 0 )
    return MINC2_ERROR;
  
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
      h->dimension_indeces[1]=i;
    }
    else if(!strcmp(name,MIyspace) || !strcmp(name,MIyfrequency) ) /*this is Y
                                                                      space */
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_Y;
      h->dimension_indeces[2]=i;
    }
    else if(!strcmp(name,MIzspace) || !strcmp(name,MIzfrequency) ) /*this is Z
                                                                    space*/
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_Z;
      h->dimension_indeces[3]=i;
    }
    else if(!strcmp(name,MIvector_dimension) ) /*this is vector space*/
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_VEC;
      h->dimension_indeces[0]=i;
    }
    else if(!strcmp(name,MItime) || !strcmp(name,MItfrequency) ) /*this is time
                                                                   space */
    {
      h->store_dims[h->ndims-i-1].id=MINC2_DIM_TIME;
      h->dimension_indeces[4]=i;
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
}


int minc2_setup_standard_order(minc2_file_handle h)
{
  int spatial_dimension=0;
  int usable_dimensions=0;
  int i;
  
  if(!h->store_dims)
  {
    /*minc file is not opened or created yet*/
    return MINC2_ERROR;
  }
  
  /*minc api uses inverse order of dimensions , fastest varying are last*/
  if(h->dimension_indeces[4]!=-1) /* have time dimension*/
  {
    h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[4]];
    /*always use positive*/
    miset_dimension_apparent_voxel_order(h->apparent_dims[usable_dimensions],MI_POSITIVE);
    
    h->representation_dims[h->ndims-1-usable_dimensions]=h->store_dims[h->ndims-1-h->dimension_indeces[4]];
    miget_dimension_separation(h->apparent_dims[usable_dimensions],MI_POSITIVE,&h->representation_dims[h->ndims-1-usable_dimensions].step);
    miget_dimension_start(h->apparent_dims[usable_dimensions],MI_POSITIVE,&h->representation_dims[h->ndims-1-usable_dimensions].start);
    
    usable_dimensions++;
  }
    
  for(i=3; i>0; i--)
  {
    if(h->dimension_indeces[i]!=-1)
    {
      h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[i]];
      /*always use positive*/
      miset_dimension_apparent_voxel_order(h->apparent_dims[usable_dimensions],MI_POSITIVE);
      
      h->representation_dims[h->ndims-1-usable_dimensions]=h->store_dims[h->ndims-1-h->dimension_indeces[i]];
      miget_dimension_separation(h->apparent_dims[usable_dimensions],MI_POSITIVE,&h->representation_dims[h->ndims-1-usable_dimensions].step);
      miget_dimension_start(h->apparent_dims[usable_dimensions],MI_POSITIVE,&h->representation_dims[h->ndims-1-usable_dimensions].start);
      
      spatial_dimension++;
      usable_dimensions++;
    }
  }

  if(h->dimension_indeces[0]!=-1) /* have vector dimension*/
  {
    h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[0]];
    /*should i set positive voxel order too, just in case?*/
    
    h->representation_dims[h->ndims-1-usable_dimensions]=h->store_dims[h->ndims-1-h->dimension_indeces[0]];
    
    usable_dimensions++;
  }

  /*Set apparent dimension order to the MINC2 api*/
  if(miset_apparent_dimension_order(h->vol,usable_dimensions,h->apparent_dims)<0)
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
    
    return _minc2_cleanup_dimensions(h);
  } else {
    /*File was not open?*/
    return MINC2_ERROR;
  }
  
}

int minc2_ndim(minc2_file_handle h,int *ndim)
{
  if(h->ndims)
  {
    *ndim=h->ndims;
  } else 
    return MINC2_ERROR;
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
  int usable_dimensions=0;
  int i;
  int err=MINC2_SUCCESS;
  misize_t *start=(misize_t *)calloc(h->ndims,sizeof(misize_t));
  misize_t *count=(misize_t *)calloc(h->ndims,sizeof(misize_t));

  if(h->using_apparent_order)
  {
    /*need to specify dimensions in apparent order, with minc2 convention that fasted dimensions are last*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      start[i]=0;
      count[i]=h->representation_dims[h->ndims-i-1].length;
    }
  } else {
    /*will read information on file order*/
    for ( i = 0; i < h->ndims ; i++ )
    {
      start[i]=0;
      count[i]=h->store_dims[h->ndims-i-1].length;;
    }
  }
  
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

  if ( miget_real_value_hyperslab(h->vol, buffer_type, start, count, buffer) < 0 )
    err=MINC2_ERROR;
  else
    err=MINC2_SUCCESS;
  
  free( start ); 
  free( count );
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
  
  h->dimension_name    = NULL;
  h->dimension_size    = NULL;
  h->dimension_start   = NULL;
  h->dimension_step    = NULL;
  h->file_dims         = NULL;
  h->apparent_dims     = NULL;
  h->store_dims        = NULL;
  h->representation_dims=NULL;
  h->using_apparent_order=0;
  
  return MINC2_SUCCESS;
}

static int _minc2_allocate_dimensions(minc2_file_handle h,int nDims)
{
  int i;
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

  for (i = 0; i < 5; i++ )
  {
    h->dimension_indeces[i] = -1;
  }
  /*TODO: check if memory was allocated?*/
  
  return MINC2_SUCCESS;
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


/* kate: indent-mode cstyle; indent-width 2; replace-tabs on; */