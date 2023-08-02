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

# torch stuff
import torch

from minc2_simple import minc2_file
from minc2_simple import minc2_xfm,minc2_dim

import torch.nn.functional as F

def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))


def hdr_to_affine(hdr):
    rot=np.zeros((3,3))
    scales=np.zeros((3,3))
    start=np.zeros(3)

    ax = np.array([h.id for h in hdr])

    for i in range(3):
        aa=np.where(ax == (i+1))[0][0] # HACK, assumes DIM_X=1,DIM_Y=2 etc
        if hdr[aa].have_dir_cos:
            rot[i,:] = hdr[aa].dir_cos
        else:
            rot[i,i] = 1

        scales[i,i] = hdr[aa].step
        start[i] = hdr[aa].start
    origin = start@rot
    out=np.eye(4)

    out[0:3,0:3] = (scales.T*rot).T
    out[0:3,3] = origin
    return out


def affine_to_dims(aff, shape):
    # convert to minc2 sampling format
    start, step, dir_cos = decompose(aff)
    if len(shape) == 3: # this is a 3D volume
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
            ]
    elif len(shape) == 4: # this is a 3D grid volume, vector space is the last one
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
            ] + [ minc2_dim(id=minc2_file.MINC2_DIM_VEC, length=shape[3], start=0, step=1, have_dir_cos=False, dir_cos=[0,0,0])]
    else:
        assert(False)
    return dims


""" 
    Load minc volume into numpy array and return voxel2wordl matrix too
"""
def load_minc_volume(fname, as_byte=False):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    d = mm.load_complete_volume_tensor(minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_DOUBLE)
    aff=np.asmatrix(hdr_to_affine(mm.representation_dims()))

    mm.close()
    return d, aff


def save_minc_volume(fn, data, aff, ref_fname=None, history=None):
    dims=affine_to_dims(aff, data.shape)
    out=minc2_file()
    out.define(dims, minc2_file.MINC2_SHORT, minc2_file.MINC2_DOUBLE)
    out.create(fn)
    if ref_fname is not None:
        ref=minc2_file(ref_fname)
        out.copy_metadata(ref)
    # if history is not None:
    #     old_history=out.read_attribute("","history")
    #     new_history=old_history+"\n"+history
    #     out.write_attribute("","history",new_history)

    out.setup_standard_order()
    out.save_complete_volume_tensor(data)
    out.close()
    
def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag(aff[0:3,0:3] @ np.linalg.inv(dir_cos))
    start = (aff[0:3,3].T @ np.linalg.inv(dir_cos)).T
    return start, step, dir_cos


def load_nl_xfm(fn):
    x=minc2_xfm(fn)
    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        assert(False)
    else:
        _identity=np.asmatrix(np.identity(4))
        _eps=1e-6
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(0)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            grid_file, grid_invert =x.get_grid_transform(0)
        else:
            # probably unsupported type
            assert(False)

        # load grid file into 4D memory
        grid, v2w = load_minc_volume(grid_file, as_byte=False)
        return grid, v2w, grid_invert


def parse_options():

    parser = argparse.ArgumentParser(description='Apply nonlinear xfms to minc file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input minc file")
    
    parser.add_argument("xfm", type=str, 
                        help="Input transform file")
    
    parser.add_argument("output", type=str, 
                        help="Output minc file")
    
    parser.add_argument("--grid", type=str, 
                        help="output grid prefix (DEBUG)")

    params = parser.parse_args()
    
    return params

# voxel to pytorch:
def create_v2p_matrix(shape):
    v2p = np.diag( [2/shape[2],   2/shape[1],   2/shape[0], 1])
    v2p[0:3,3] = (  1/shape[2]-1, 1/shape[1]-1, 1/shape[0]-1  ) # adjust for half a voxel shift
    return v2p

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    data, v2w_data = load_minc_volume(params.input)
    grid_xfm, v2w_xfm, inv_xfm = load_nl_xfm(params.xfm)

    v2p_data = create_v2p_matrix(data.shape)
    v2p_grid = create_v2p_matrix(grid_xfm.shape)

    # matrix that allows to resample grid_xfm into data space
    grid_xfm_matrix = v2p_grid @ np.linalg.inv(v2w_xfm) @ v2w_data @ np.linalg.inv(v2p_data)
    print(f"{grid_xfm_matrix[0:3,0:4]=}")
    map_grid = F.affine_grid(torch.tensor(grid_xfm_matrix[0:3,0:4]).unsqueeze(0), [1, 1, *data.size()], align_corners=True)

    # add identity transform
    identity_grid = F.affine_grid(torch.tensor(np.eye(4)[0:3,0:4]).unsqueeze(0), [1, 1, *data.size()], align_corners=True)
    # resample grid into data space

    # and convert to torch sampling (-1,1)
    # here we assume that there is no rotation that need to be applied (!)
    # otherwise we would need to apply each voxel vector to a rotaion matrix
    _,step,_ = decompose(v2w_data)
    voxel_scaling_factor = (torch.tensor([2.0/data.shape[2], 2.0/data.shape[1], 2.0/data.shape[0]]) / torch.tensor(step) ).unsqueeze(0).unsqueeze(0)

    pytorch_grid = torch.stack([ 
        F.grid_sample(grid_xfm[:,:,:,i].unsqueeze(0).unsqueeze(0), map_grid, align_corners=True).squeeze(0).squeeze(0)*2
                       for i in range(3) ],dim=3).unsqueeze(0) * voxel_scaling_factor

    if params.grid is not None:
        for i in range(3):
            save_minc_volume( f"{params.grid}_pytorch_{i}.mnc", 
                                pytorch_grid[0,:,:,:,i].contiguous(), v2w_data, ref_fname=params.input, history=_history)

    # finally apply transformation (need to add identity transform)
    out = F.grid_sample(data.unsqueeze(0).unsqueeze(0), pytorch_grid+ identity_grid,
                        align_corners=True).squeeze(0).squeeze(0)

    print("Will save to "+params.output)
    save_minc_volume( params.output,out,v2w_data, ref_fname=params.input, history=_history)