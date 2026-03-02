#include "minc2-simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int error_cnt = 0;

#define TESTRPT(msg, val) (error_cnt++, fprintf(stderr, \
                           "Error reported on line #%d, %s: %d\n", \
                           __LINE__, msg, val))

int main(int argc,char **argv)
{
  minc2_file_handle h;
  int ndims;
  int dims[10];
  int var_type;
  double *time_steps=NULL;
  int start[]={0};
  int count[1];

  if(argc<2)
  {
    fprintf(stderr,"Usage: %s <input_4d.mnc>\n",argv[0]);
    return 1;
  }

  h=minc2_allocate0();
  if(!h)
  {
    TESTRPT("failed to allocate handle",0);
    return error_cnt;
  }

  if(minc2_open(h,argv[1])!=MINC2_SUCCESS)
  {
    TESTRPT("failed to open file",0);
    minc2_free(h);
    return error_cnt;
  }

  /* test minc2_get_variable_ndims for time dimension */
  if(minc2_get_variable_ndims(h,"dimensions","time",&ndims)!=MINC2_SUCCESS)
  {
    TESTRPT("failed to get variable ndims for dimensions/time",0);
  } else {
    fprintf(stdout,"dimensions/time ndims=%d\n",ndims);
    if(ndims!=1)
    {
      TESTRPT("unexpected ndims for time variable",ndims);
    }
  }

  /* test minc2_get_variable_dims for time dimension */
  if(minc2_get_variable_dims(h,"dimensions","time",dims)!=MINC2_SUCCESS)
  {
    TESTRPT("failed to get variable dims for dimensions/time",0);
  } else {
    fprintf(stdout,"dimensions/time dims[0]=%d\n",dims[0]);
    if(dims[0]!=3)
    {
      TESTRPT("unexpected dim size for time variable",dims[0]);
    }
  }

  /* test minc2_get_variable_type for time dimension */
  if(minc2_get_variable_type(h,"dimensions","time",&var_type)!=MINC2_SUCCESS)
  {
    TESTRPT("failed to get variable type for dimensions/time",0);
  } else {
    fprintf(stdout,"dimensions/time type=%s\n",minc2_data_type_name(var_type));
    if(var_type!=MINC2_DOUBLE)
    {
      TESTRPT("unexpected type for time variable",var_type);
    }
  }

  /* test minc2_read_variable_raw for time dimension */
  count[0]=dims[0];
  time_steps=(double*)malloc(dims[0]*sizeof(double));
  if(!time_steps)
  {
    TESTRPT("failed to allocate memory for time steps",0);
  } else {
    if(minc2_read_variable_raw(h,"dimensions","time",MINC2_DOUBLE,start,count,time_steps)!=MINC2_SUCCESS)
    {
      TESTRPT("failed to read variable raw for dimensions/time",0);
    } else {
      int i;
      fprintf(stdout,"Time steps:");
      for(i=0;i<dims[0];i++)
        fprintf(stdout," %f",time_steps[i]);
      fprintf(stdout,"\n");

      /* verify expected values: 0, 1, 10 */
      if(fabs(time_steps[0]-0.0)>1e-6)
      {
        TESTRPT("unexpected time_steps[0]",(int)time_steps[0]);
      }
      if(fabs(time_steps[1]-1.0)>1e-6)
      {
        TESTRPT("unexpected time_steps[1]",(int)time_steps[1]);
      }
      if(fabs(time_steps[2]-10.0)>1e-6)
      {
        TESTRPT("unexpected time_steps[2]",(int)time_steps[2]);
      }
    }
    free(time_steps);
    time_steps=NULL;
  }

  /* test that querying a nonexistent variable returns error */
  if(minc2_get_variable_ndims(h,"dimensions","nonexistent",&ndims)==MINC2_SUCCESS)
  {
    TESTRPT("expected error for nonexistent variable but got success",0);
  }

  minc2_close(h);
  minc2_free(h);

  if(error_cnt!=0)
  {
    fprintf(stderr,"%d error%s reported\n",
            error_cnt,(error_cnt==1)?"":"s");
  } else {
    fprintf(stderr,"No errors\n");
  }

  return error_cnt;
}

/* kate: indent-mode cstyle; indent-width 2; replace-tabs on; remove-trailing-spaces modified; hl c*/
