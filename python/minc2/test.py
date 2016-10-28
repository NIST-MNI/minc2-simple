from minc2.simple import minc2_file
import sys


if __name__ == "__main__":
    m=minc2_file("/opt/minc/share/icbm152_model_09c/mni_icbm152_t1_tal_nlin_sym_09c.mnc")
    print("dims={}".format(m.ndim()))
    m.close()
    