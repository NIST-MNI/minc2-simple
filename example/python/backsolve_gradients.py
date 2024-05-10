#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 29/01/2018

import argparse
import re
from time import gmtime, strftime
import sys

import numpy as np
import time
#from minc2_simple import minc2_file
#from minc2_simple import minc2_xfm,minc2_dim

import scipy.sparse as sp

from minc.io import *
#from minc.geo import *


def parse_options():

    parser = argparse.ArgumentParser(description='Apply xfms to minc file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input 4D grid file (gradients)")
    
    parser.add_argument("output", type=str, 
                        help="Output 3D minc file")
    
    parser.add_argument('--forward', action="store_true",
                        default=False,
                        help='Use forward difference (shifted by half voxel)' )
    
    params = parser.parse_args()
    
    return params



if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    # data - 4D tensor, v2w - voxel to world matrix
    # data will have shap X,Y,Z,3
    t1=time.time()
    data, v2w = load_minc_volume_np(params.input,dtype='float64') # volume andvoxel to world matrix

    # extrac voxel step size 
    _, step, _ = decompose(v2w)
    print(f"{step=}")

    t2=time.time()
    print(f"Loading time: {t2-t1} sec")

    strides=np.array([data.shape[1]*data.shape[2], data.shape[2], 1] ,dtype=np.int64)
    nvox=np.prod(data.shape[0:3])

    b0=np.zeros(3,dtype=np.int64)
    b1=np.array(data.shape[0:3])-1

    # converts i,j,k index to a linear index, clipping at the edges
    #ijk2idx = lambda ijk: np.dot( np.minimum(np.maximum(np.array(ijk), b0), b1), strides)
    # faster version, when we are ceratain that edges will not be exceeded
    ijk2idx = lambda ijk: np.dot( np.array(ijk), strides)

    # need to setup system of sparse equations
    # this loop is a very simple one and can be optimized
    # 1. for each voxel, we have 3 values
    data_dx=np.zeros(nvox*2)
    data_dy=np.zeros(nvox*2)
    data_dz=np.zeros(nvox*2)

    row_dx=np.zeros(nvox*2)
    col_dx=np.zeros(nvox*2)

    row_dy=np.zeros(nvox*2)
    col_dy=np.zeros(nvox*2)

    row_dz=np.zeros(nvox*2)
    col_dz=np.zeros(nvox*2)

    dx,dy,dz=step.tolist()

    # we are cheating on the edges , replacing centered differences with forward/backward, but with wrong argument
    print("Filling the matrix")
    if not params.forward: 
        for i in range(1,data.shape[0]-1):
            for j in range(1,data.shape[1]-1):
                for k in range(1,data.shape[2]-1):

                    idx=ijk2idx([i,j,k])
                    # in minc world , the indexes are actually reverse, i->z, j->y, k->x

                    #  df/dx ~ (f(x+1,y,z)-f(x-1,y,z))/2
                    row_dx [idx*2]=idx
                    col_dx [idx*2]=ijk2idx([i,j,k-1])
                    data_dx[idx*2]=-0.5/dx

                    row_dx [idx*2+1]=idx
                    col_dx [idx*2+1]=ijk2idx([i,j,k+1])
                    data_dx[idx*2+1]=0.5/dx
                    #####
                    #  df/dy ~ (f(x,y+1,z)-f(x,y-1,z))/2
                    row_dy [idx*2]=idx+nvox
                    col_dy [idx*2]=ijk2idx([i,j-1,k])
                    data_dy[idx*2]=-0.5/dy

                    row_dy [idx*2+1]=idx+nvox
                    col_dy [idx*2+1]=ijk2idx([i,j+1,k])
                    data_dy[idx*2+1]=0.5/dy
                    #####
                    #  df/dz ~ (f(x,y,z+1)-f(x,y,z-1))/2
                    row_dz [idx*2]=idx+nvox*2
                    col_dz [idx*2]=ijk2idx([i-1,j,k])
                    data_dz[idx*2]=-0.5/dz

                    row_dz [idx*2+1]=idx+nvox*2
                    col_dz [idx*2+1]=ijk2idx([i+1,j,k])
                    data_dz[idx*2+1]=0.5/dz
    else: # forward difference, shifted by half voxel 
        for i in range(data.shape[0]-1):
            for j in range(data.shape[1]-1):
                for k in range(data.shape[2]-1):

                    idx=ijk2idx([i,j,k])
                    # in minc world , the indexes are actually reverse, i->z, j->y, k->x

                    #  df/dx ~ (f(x+1,y,z)-f(x-1,y,z))/2
                    row_dx [idx*2]=idx
                    col_dx [idx*2]=idx
                    data_dx[idx*2]=-1./dx

                    row_dx [idx*2+1]=idx
                    col_dx [idx*2+1]=ijk2idx([i,j,k+1])
                    data_dx[idx*2+1]=1./dx  
                    #####
                    #  df/dy ~ (f(x,y+1,z)-f(x,y-1,z))/2
                    row_dy [idx*2]=idx+nvox
                    col_dy [idx*2]=idx
                    data_dy[idx*2]=-1./dy

                    row_dy [idx*2+1]=idx+nvox
                    col_dy [idx*2+1]=ijk2idx([i,j+1,k])
                    data_dy[idx*2+1]=1./dy
                    #####
                    #  df/dz ~ (f(x,y,z+1)-f(x,y,z-1))/2
                    row_dz [idx*2]=idx+nvox*2
                    col_dz [idx*2]=idx
                    data_dz[idx*2]=-1./dz

                    row_dz [idx*2+1]=idx+nvox*2
                    col_dz [idx*2+1]=ijk2idx([i+1,j,k])
                    data_dz[idx*2+1]=1./dz

    t3=time.time()
    print(f"Matrix fill time: {t3-t2} sec")
    print("Converting to CSR")

    mtx=np.concatenate((data_dx,data_dy,data_dz) ) #  data for design matrix
    rows=np.concatenate((row_dx, row_dy, row_dz) )
    cols=np.concatenate((col_dx, col_dy, col_dz) )

    # left size, finite-difference design matrix
    A=sp.coo_array( (mtx,( rows, cols ) ),
                     shape=(nvox*3,
                            nvox) ).tocsr()
    
    # # right side dX,dY,dZ
    B=np.concatenate((data[:,:,:,0].flatten(), 
                      data[:,:,:,1].flatten(), 
                      data[:,:,:,2].flatten(),
                      #np.zeros(data.shape[1]*data.shape[2])
                      ))
    t4=time.time()
    print(f"Sparse matrix conversion setup time: {t4-t3} sec")
    #A = sp.coo_array((data_dx,(row_dx,col_dx)), shape=(nvox,nvox)).tocsr()
    #B = data[:,:,:,0].flatten()

    print("Solving...")
    # solve the system, in a least squared fashion
    #X, istop, itn, r1norm, r2norm,anorm, acond, arnorm,xnorm,var =sp.linalg.lsqr(A,B,show=True,)
    X =sp.linalg.lsmr(A,B,show=True,atol=1e-8,btol=1e-8)[0]
    t5=time.time()
    print(f"Solving time: {t5-t4} sec")
    
    out=np.reshape(X,data.shape[0:3])

    save_minc_volume( params.output, out , v2w, history=_history)
