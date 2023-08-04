from time import gmtime, strftime

from minc2_simple import minc2_file
from minc2_simple import minc2_xfm,minc2_dim

import numpy as np


from .geo import decompose


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


def load_lin_xfm(fn):
    _identity=np.asmatrix(np.identity(4))
    _eps=1e-6
    x=minc2_xfm(fn)

    if x.get_n_concat()==1 and x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR:
        # this is a linear matrix
        lin_xfm=np.asmatrix(x.get_linear_transform())
        return lin_xfm
    else:
        if x.get_n_type(0)==minc2_xfm.MINC2_XFM_LINEAR and x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # is this identity matrix
            assert(np.linalg.norm(_identity-np.asmatrix(x.get_linear_transform(0)) )<=_eps)
            # TODO: if grid have to be inverted!
            grid_file, grid_invert=x.get_grid_transform(1)
        elif x.get_n_type(1)==minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
            # TODO: if grid have to be inverted!
            (grid_file, grid_invert)=x.get_grid_transform(0)
        assert(False) # TODO
        return None