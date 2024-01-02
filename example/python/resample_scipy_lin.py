#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 2024-01-02

import argparse
import re
from time import gmtime, strftime
import sys

import numpy as np

from minc2_simple import minc2_file

from minc.io import load_lin_xfm, hdr_to_affine, affine_to_dims, format_history
from minc.geo import *

import scipy


def load_minc_volume(fname, as_byte=False):
    mm=minc2_file(fname)
    mm.setup_standard_order()

    d = mm.load_complete_volume(minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_DOUBLE)
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
    out.save_complete_volume(data)
    out.close()



def parse_options():

    parser = argparse.ArgumentParser(description='Apply xfms to minc file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input minc file")
    
    parser.add_argument("xfm", type=str, 
                        help="Input transform file (linear only))")
    
    parser.add_argument("output", type=str, 
                        help="Output minc file")
    
    parser.add_argument("--like", type=str, 
                        help="Use this sampling for output")
    
    parser.add_argument("--order", type=int, default=1, 
                        help="Resample order")
    
    parser.add_argument("--fill", type=float, default=0.0, 
                        help="Fill value")

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    data, v2w = load_minc_volume(params.input) # volume and voxel to world matrix

    if params.like is not None:
        like_data, like_v2w = load_minc_volume(params.like)
    else:
        like_data = data
        like_v2w = v2w

    print(f"{data.shape=} {like_data.shape=}")

    xfm = load_lin_xfm(params.xfm) # transformation matrix

    # world to voxel matrix
    w2v = np.linalg.inv(v2w)

    # voxel storage matrix
    xyz_to_zyx = np.array([[0,0,1,0],
                           [0,1,0,0],
                           [1,0,0,0],
                           [0,0,0,1]])

    # full transformation matrix in voxel space
    full_xfm = w2v @ np.linalg.inv(xfm) @ like_v2w 
    print(f"{full_xfm=}")

    # now need to change the dimension order, because in minc world it is XYZ and in scipy it is ZYX
    full_xfm = xyz_to_zyx@full_xfm@xyz_to_zyx

    # debug 
    out = scipy.ndimage.affine_transform(data, full_xfm, output_shape=like_data.shape, order=params.order,mode='constant',cval=params.fill)
    print(f"{out.shape=}")
    
    #out = F.grid_sample(data.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze(0).squeeze(0)

    print("Will save to "+params.output)
    save_minc_volume( params.output, out , like_v2w, ref_fname=params.input,history=_history)
    