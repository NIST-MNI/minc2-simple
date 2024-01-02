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

    parser = argparse.ArgumentParser(description='Uniformize minc volume to 1x1x1 step size',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input minc file")
    
    parser.add_argument("output", type=str, 
                        help="Output minc file")
    
    parser.add_argument("--order", type=int, default=1, 
                        help="Resample order")
    
    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    data, v2w = load_minc_volume(params.input) # volume and voxel to world matrix
    start, step, dir_cos = decompose(v2w)

    # voxel storage matrix
    xyz_to_zyx = np.array([[0,0,1,0],
                           [0,1,0,0],
                           [1,0,0,0],
                           [0,0,0,1]])
    
    # need to account for the different order of dimensions
    new_shape = np.ceil(np.array(data.shape) * step[[2,1,0]]).astype(int)

    # have to account for the shift of the voxel center
    new_start = start - step*0.5 + np.ones(3)*0.5

    new_v2w = compose(new_start, np.ones(3), dir_cos)

    full_xfm = xyz_to_zyx @ np.linalg.inv(v2w) @ new_v2w @ xyz_to_zyx

    out = scipy.ndimage.affine_transform(data, full_xfm, output_shape=new_shape, order=params.order, mode='constant',cval=0.0)

    save_minc_volume( params.output, out , new_v2w, ref_fname=params.input, history=_history)
    