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

# deep models
import torch
from model.util import *

# debug
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
    ax = np.array([hdr[j].id for j in range(3)])

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

def affine_to_dims(aff,shape):
    start, step, dir_cos = decompose(aff)
    dims=[
        minc2_dim(id=i+1,length=shape[i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
    ]
    return dims


def load_input(fname, as_byte=False):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    d = mm.load_complete_volume_tensor(minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_DOUBLE)
    aff=np.asmatrix(hdr_to_affine(mm.representation_dims()))

    mm.close()
    return d, aff


def save_minc_volume(fn,data,aff,ref_fname=None,history=None):
    dims=affine_to_dims(aff,data.shape)
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

def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag(aff[0:3,0:3] @ np.linalg.inv(dir_cos))
    start = (aff[0:3,3].T @ np.linalg.inv(dir_cos)).T
    return start, step, dir_cos


def load_xfm(fn):
    x=minc2_xfm(fn)
    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        # this is a linear matrix
        lin_xfm=np.asmatrix(x.get_linear_transform())
        return lin_xfm
    else:
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            assert(scipy.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            # TODO: if grid have to be inverted!
            grid_file,grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            (grid_file,grid_invert)=x.get_grid_transform(0)
        assert(False) # TODO
        return None


def parse_options():

    parser = argparse.ArgumentParser(description='Apply xfms to minc file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input minc file")
    
    parser.add_argument("xfm", type=str, 
                        help="Input transform file")
    
    parser.add_argument("output", type=str, 
                        help="Output minc file")

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    data, v2w=load_input(params.input)
    xfm=load_xfm(params.xfm)
    w2v=np.linalg.inv(v2w)
    # TODO: save
    print(f"{v2w=}\n{w2v=}\n{xfm=}\n{data.shape=}\n{data.dtype=}")

    # voxel to pytorch:
    v2p = np.diag( [2/data.shape[0],  2/data.shape[1], 2/data.shape[2], 1])
    v2p[0:3,3] = (1/data.shape[0]-1, 1/data.shape[1]-1, 1/data.shape[2]-1) # adjust for half a voxel shift

    p2v=np.linalg.inv(v2p)

    print(f"{v2p=}\n{p2v=}")

    full_xfm=v2p @ w2v @ np.linalg.inv(xfm) @ v2w @ p2v

    print(f"{full_xfm[0:3,0:4]=}")

    grid = F.affine_grid(torch.tensor(full_xfm[0:3,0:4]).unsqueeze(0), [1, 1, *data.size()],align_corners=True)

    print(f"{grid.shape=} {grid.dtype=}")
    out = F.grid_sample(data.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze(0).squeeze(0)

    print(f"{out.shape=}")

    print("Will save to "+params.output)
    save_minc_volume( params.output,out,v2w, ref_fname=params.input,history=_history)