#include "minc2.h"

#include "minc2-simple.h"
#include <stdlib.h>
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

 
  midimhandle_t *file_dims;
  midimhandle_t *apparent_dims;
  
  miboolean_t slice_scaling_flag;
  miboolean_t global_scaling_flag;
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
  
  if ( miget_dimension_sizes(h->file_dims, h->ndims, h->dimension_size) < 0 )
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

  /*setup normalized representation of data (if needed only)*/
  for (i = 0; i < h->ndims; i++ )
  {
    char *      name;
    double      _sep;
    /*const char *_sign="+";*/

    if ( miget_dimension_name(h->file_dims[i], &name) < 0 )
      return MINC2_ERROR;

    h->dimension_name[i] = name;
    
    if(!strcmp(name,MIxspace) || !strcmp(name,MIxfrequency) ) /*this is X space*/
      {
      h->dimension_indeces[1]=i;
      }
    else if(!strcmp(name,MIyspace) || !strcmp(name,MIyfrequency) ) /*this is Y
                                                                      space */
      {
      h->dimension_indeces[2]=i;
      }
    else if(!strcmp(name,MIzspace) || !strcmp(name,MIzfrequency) ) /*this is Z
                                                                    space*/
      {
      h->dimension_indeces[3]=i;
      }
    else if(!strcmp(name,MIvector_dimension) ) /*this is vector space*/
      {
      h->dimension_indeces[0]=i;
      }
    else if(!strcmp(name,MItime) || !strcmp(name,MItfrequency) ) /*this is time
                                                                   space */
      {
      h->dimension_indeces[4]=i;
      }
    else
      {
        MI_LOG_ERROR(MI2_MSG_GENERIC,"Unsupported dimension type:%s",name);
        return MINC2_ERROR;
      }
  }

  for(i=1; i<4; i++)
    {
    if(h->dimension_indeces[i]!=-1) /*this dimension is present*/
      {
      spatial_dimension_count++;
      }
    }

  if ( spatial_dimension_count == 0 ) /* sorry, this is metaphysical question*/
    {
    MI_LOG_ERROR(MI2_MSG_GENERIC,"No spatial dimensions found");
    return MINC2_ERROR;
    }

  if ( h->dimension_indeces[0]!=-1 && h->dimension_indeces[4]!=-1 )
    {
      MI_LOG_ERROR(MI2_MSG_GENERIC,"time + vector dimensions are not supported simultaneously");
      return MINC2_ERROR; /*time + vector dimension not supported right now*/
    }
  
  /*minc api uses inverse order of dimensions , fastest varying are last*/
  for(i=3; i>0; i--)
    {
    if(h->dimension_indeces[i]!=-1)
      {
      misize_t _sz;
    
      h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[i]];
      //always use positive
      miset_dimension_apparent_voxel_order(h->apparent_dims[usable_dimensions],MI_POSITIVE);
      miget_dimension_size(h->apparent_dims[usable_dimensions],&_sz);

      /* 
      std::vector< double > _dir(3);
      double                _sep,_start;

      miget_dimension_separation(h->m_MincApparentDims[usable_dimensions],MI_ORDER_APPARENT,&_sep);
      miget_dimension_cosines(h->m_MincApparentDims[usable_dimensions],&_dir[0]);
      miget_dimension_start(h->m_MincApparentDims[usable_dimensions],MI_ORDER_APPARENT,&_start);

      for(int j=0; j<3; j++)
        dir_cos[j][i-1]=_dir[j];

      origin[i-1]=_start;
      sep[i-1]=_sep;

      h->SetDimensions(i-1,static_cast<unsigned int>(_sz) );
      h->SetDirection(i-1,_dir);
      h->SetSpacing(i-1,_sep);*/

      spatial_dimension++;
      usable_dimensions++;
      }
    }

  if(h->dimension_indeces[0]!=-1) /* have vector dimension*/
    {
    misize_t _sz;
    h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[0]];
  
    miget_dimension_size(h->apparent_dims[usable_dimensions],&_sz);
    /*numberOfComponents=_sz;*/
    usable_dimensions++;
    }

  if(h->dimension_indeces[4]!=-1) /* have time dimension*/
    {
    misize_t _sz;
    h->apparent_dims[usable_dimensions]=h->file_dims[h->dimension_indeces[4]];
    /*always use positive*/
    miset_dimension_apparent_voxel_order(h->apparent_dims[usable_dimensions],MI_POSITIVE);
    miget_dimension_size(h->apparent_dims[usable_dimensions],&_sz);
    /*numberOfComponents=_sz;*/
    usable_dimensions++;
    }

  /*Set apparent dimension order to the MINC2 api*/
  if(miset_apparent_dimension_order(h->vol,usable_dimensions,h->apparent_dims)<0)
    return MINC2_ERROR;  


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

int minc2_load_complete_volume( minc2_file_handle h,void *buffer,int representation_type)
{
  mitype_t buffer_type=MI_TYPE_UBYTE;
  int usable_dimensions=0;
  int i;
  misize_t *start=(misize_t *)calloc(h->ndims,sizeof(misize_t));
  misize_t *count=(misize_t *)calloc(h->ndims,sizeof(misize_t));

  /*need to specify dimensions in apparent order*/
  for ( i = 0; i < 5; i++ )
  {
    if(h->dimension_indeces[i]>=0)
    {
      start[usable_dimensions]=0;
      count[usable_dimensions]=h->dimension_size[h->dimension_indeces[i]];
      usable_dimensions++;
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
  {
    free( start ); 
    free( count );
    return MINC2_ERROR;
  }
  free( start ); 
  free( count );
  return MINC2_SUCCESS;
}

static int _minc2_cleanup_dimensions(minc2_file_handle h)
{
  int i;
  if( h->dimension_name )
    {
    for ( i = 0; i < h->ndims; i++ )
      {
      mifree_name( h->dimension_name[i] );
      h->dimension_name[i]=NULL;
      }
    }

  free(h->dimension_name);
  free(h->dimension_size);
  free(h->dimension_start);
  free(h->dimension_step);
  free(h->file_dims);
  free(h->apparent_dims);

  h->dimension_name    = NULL;
  h->dimension_size    = NULL;
  h->dimension_start   = NULL;
  h->dimension_step    = NULL;
  h->file_dims         = NULL;
  h->apparent_dims     = NULL;
  return MINC2_SUCCESS;
}

static int _minc2_allocate_dimensions(minc2_file_handle h,int nDims)
{
  int i;
  _minc2_cleanup_dimensions(h);

  h->ndims=nDims;

  h->dimension_name  = (char**)calloc(h->ndims,sizeof(char*));
  h->dimension_size  = calloc(h->ndims,sizeof(misize_t));
  h->dimension_start = calloc(h->ndims,sizeof(double));
  h->dimension_step  = calloc(h->ndims,sizeof(double));
  h->file_dims       = calloc(h->ndims,sizeof(midimhandle_t));
  h->apparent_dims   = calloc(h->ndims,sizeof(midimhandle_t));

  for (i = 0; i < 5; i++ )
  {
    h->dimension_indeces[i] = -1;
  }
  /*TODO: check if memory was allocated?*/
  
  return MINC2_SUCCESS;

}
