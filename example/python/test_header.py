from minc2_simple import minc2_file,minc2_dim
import sys
import numpy as np


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


def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag(aff[0:3,0:3] @ np.linalg.inv(dir_cos))
    start = (aff[0:3,3].T @ np.linalg.inv(dir_cos)).T
    return start, step, dir_cos

def affine_to_dims(aff,shape):
    start, step, dir_cos = decompose(aff)
    dims=[
        minc2_dim(id=i+1,length=shape[i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=dir_cos[i,0:3]) for i in range(3)
    ]
    return dims



def load_minc_volume(fn):
    m=minc2_file(fn)
    m.setup_standard_order()
    data=m.load_complete_volume(minc2_file.MINC2_FLOAT)
    aff=hdr_to_affine(m.representation_dims())

    return data,aff

def save_minc_volume(fn,data,aff):
    dims=affine_to_dims(aff,data.shape)

    out=minc2_file()
    out.define(dims, minc2_file.MINC2_SHORT, minc2_file.MINC2_FLOAT)
    out.create(fn)
    #out.copy_metadata(ref)
    out.setup_standard_order()
    out.save_complete_volume(data)
    

if __name__ == "__main__":
    infile='/opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc'

    data,aff=load_minc_volume(infile)
    print(f"{aff=}")
    start, step, dir_cos = decompose(aff)
    print(f"{start=} {step=} {dir_cos=}")

    save_minc_volume("test.mnc",data,aff)
    