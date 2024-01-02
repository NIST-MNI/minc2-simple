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

from minc.io import *
from minc.geo import *


def parse_options():

    parser = argparse.ArgumentParser(description='Apply xfms to minc file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=str, default=None,
                        help="Input minc file")
    
    parser.add_argument("xfm", type=str, 
                        help="Input transform file")
    
    parser.add_argument("output", type=str, 
                        help="Output minc file")
    
    parser.add_argument("--grid", type=str, 
                        help="output grid prefix")
    
    parser.add_argument("--like", type=str, 
                        help="Use this sampling for output")

    params = parser.parse_args()
    
    return params

if __name__ == '__main__':
    _history=format_history(sys.argv)
    params = parse_options()

    data, v2w = load_minc_volume(params.input) # volume andvoxel to world matrix

    if params.like is not None:
        like_data, like_v2w = load_minc_volume(params.like)
    else:
        like_data = data
        like_v2w = v2w

    xfm = load_lin_xfm(params.xfm) # transformation matrix

    # world to voxel matrix
    w2v = np.linalg.inv(v2w)

    # convert from voxel notation to pytorch notation
    full_xfm = create_v2p_matrix(data.shape) @ w2v @ np.linalg.inv(xfm) @ like_v2w @ np.linalg.inv(create_v2p_matrix(like_data.shape) )

    grid_size = [1, 1, *like_data.shape]
    # convert into deformation field
    grid = F.affine_grid(torch.tensor(full_xfm[0:3, 0:4]).unsqueeze(0), 
                         grid_size, align_corners=False)

    # debug 
    if params.grid is not None:
        for i in range(3):
            save_minc_volume( f"{params.grid}_{i}.mnc", grid[0,:,:,:,i].contiguous(), like_v2w, ref_fname=params.input, history=_history)
    
    out = F.grid_sample(data.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze(0).squeeze(0)

    print("Will save to "+params.output)
    save_minc_volume( params.output, out , like_v2w, ref_fname=params.input,history=_history)
    