#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 29/01/2018

import argparse
import re
import sys

import numpy as np

# torch stuff
import torch

import torch.nn.functional as F
from minc.io import *
from minc.geo import *


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