# minc2-simple Usage Guide

A simplified C facade over libminc with Python (CFFI) and Lua bindings.
Supports reading and writing MINC2 (HDF5-based) neuroimaging volumes, transforms, and tag files.

## Table of Contents

- [Python Quick Start](#python-quick-start)
- [Python API Reference](#python-api-reference)
- [C API Reference](#c-api-reference)
- [Common Patterns](#common-patterns)
- [Key Concepts](#key-concepts)
- [Building](#building)

---

## Python Quick Start

```python
from minc2_simple import minc2_file
import numpy as np

# Read a volume
m = minc2_file('input.mnc')
data = m.data           # numpy array, shape (Z, Y, X) in standard order
step = m.step           # voxel spacings (Z, Y, X)
start = m.start         # world coordinates of first voxel
m.close()

# Write a volume based on an existing template
o = minc2_file()
o.imitate('template.mnc')
o.create('output.mnc')
o.data = data * 2.0
o.close()
```

---

## Python API Reference

### minc2_file

Primary class for reading and writing MINC2 volumes.

#### Construction and lifecycle

```python
minc2_file(path=None, standard=False, handle=None)
```
- `path` — open this file immediately on construction
- `standard` — call `setup_standard_order()` after open

```python
.open(path)           # open read-only
.open_rdwr(path)      # open for read-write (existing file)
.close()              # flush and close (required before re-reading)

.define(dims, store_type=None, representation_type=None,
        slice_scaling=None, global_scaling=None, path=None)
    # Define new volume geometry. Call before .create().
    # dims: list of minc2_dim named tuples or dicts

.create(path)         # write file header to disk (after .define())

.imitate(another, store_type=None, representation_type=None, path=None)
    # Define new volume with the same geometry as another.
    # another: minc2_file object or path string.
    # Call .create(path) afterward.
```

#### Dimension and shape queries

```python
.ndim()               # -> int: number of dimensions
.shape                # -> tuple: (dim0_len, dim1_len, ...) in standard order
.start                # -> tuple: world coordinate of first voxel per dim
.step                 # -> tuple: voxel spacing per dim (can be negative)
.store_dims()         # -> list[minc2_dim]: dimensions in file (storage) order
.representation_dims()# -> list[minc2_dim]: dimensions in memory order
.store_dtype()        # -> str: numpy dtype name for disk storage
.representation_dtype()# -> str: numpy dtype name for in-memory representation
```

`minc2_dim` is a named tuple: `(id, length, start, step, have_dir_cos, dir_cos)`
- `id` — one of `MINC2_DIM_X/Y/Z/TIME/VEC`
- `dir_cos` — numpy array of 3 direction cosines (if `have_dir_cos` is true)

#### Standard order

MINC files store dimensions in file-specific order; steps can be negative.
`setup_standard_order()` reorders to **TIME → Z → Y → X → VEC** (slowest to fastest),
flips axes to positive steps, and adjusts `start` accordingly.
All data I/O methods below operate in standard order after this call.

```python
.setup_standard_order()  # call once after open/create, before I/O
```

The `.data` and `.tensor` properties call this automatically.

#### Complete volume I/O

```python
.load_complete_volume(data_type=None)   # -> numpy.ndarray
.save_complete_volume(buf)              # buf: numpy.ndarray

# Convenience properties (call setup_standard_order internally)
.data                # get: load_complete_volume(); set: save_complete_volume()
.get_data()          # setup_standard_order() + load_complete_volume()
.set_data(arr)       # setup_standard_order() + save_complete_volume(arr)

# PyTorch variants
.load_complete_volume_tensor(data_type=None)  # -> torch.Tensor
.save_complete_volume_tensor(buf)
.tensor                                        # property (get/set)
.get_tensor()
.set_tensor(t)
```

`data_type` — one of the `MINC2_*` type constants; defaults to the file's
representation type. The return array `dtype` reflects the chosen type.

#### Hyperslab (partial) I/O

Read or write a rectangular sub-region without loading the whole volume.

```python
.load_hyperslab(slab=None, data_type=None)   # -> numpy.ndarray
.save_hyperslab(buf, start=None)

# slab / start format: tuple of per-dim (start, stop) or (start,) tuples
# Example: read first 10 slices along first dim
data = m.load_hyperslab(slab=((0, 10), (None, None), (None, None)))

# numpy-style indexing shorthand
data = m.vol[0:10, :, :]         # calls load_hyperslab
m.vol[0:10, :, :] = data         # calls save_hyperslab
```

Dimensions with a scalar index (not a slice) are squeezed from the result.

#### Coordinate transformations

```python
.voxel_to_world(ijk)   # ijk: array [i,j,k] or (n,3) -> [x,y,z] or (n,3)
.world_to_voxel(xyz)   # xyz: array [x,y,z] or (n,3) -> [i,j,k] or (n,3)
```

Coordinates are in the space defined by `start`, `step`, and `dir_cos`.

#### Scaling

```python
.set_volume_range(vmin, vmax)  # set intensity range for global scaling
```

Call before `create()` when storing integers with global scaling.
For slice-by-slice scaling, pass `slice_scaling=True` to `define()`.

#### Metadata / attributes

MINC2 metadata lives in HDF5 groups (root = `""`, others e.g. `"acquisition"`).

```python
.read_attribute(group, attribute)          # -> str, number, or np.ndarray
.write_attribute(group, attribute, value)  # value: str, bytes, or np.ndarray
.metadata()                                # -> dict[group][attr] = value
.write_metadata(d)                         # write complete metadata dict
.copy_metadata(src)                        # copy all metadata from src
```

Common attributes: `("", "history")`, `("patient", "age")`, `("acquisition", "repetition_time")`.

#### Variable / dataset access (HDF5)

Access arbitrary HDF5 datasets directly, e.g. the `time` coordinate axis.

```python
.variable_ndims(path, name)         # -> int
.variable_dims(path, name)          # -> list[int]
.variable_type(path, name)          # -> int (MINC2_* type constant)
.read_variable(path, name, data_type=None, start=None, count=None)  # -> np.ndarray
.write_variable(path, name, data, data_type=None, start=None, count=None)

# Example: read time axis coordinates from a 4D volume
t = m.read_variable("dimensions", "time")
```

`path` is the HDF5 group name; `name` is the dataset within that group.

#### Constants

```python
# Dimension types
minc2_file.MINC2_DIM_X, MINC2_DIM_Y, MINC2_DIM_Z
minc2_file.MINC2_DIM_TIME, MINC2_DIM_VEC

# Storage / representation types
minc2_file.MINC2_BYTE     # int8
minc2_file.MINC2_SHORT    # int16
minc2_file.MINC2_INT      # int32
minc2_file.MINC2_FLOAT    # float32
minc2_file.MINC2_DOUBLE   # float64
minc2_file.MINC2_UBYTE    # uint8
minc2_file.MINC2_USHORT   # uint16
minc2_file.MINC2_UINT     # uint32

# Type↔numpy mappings
minc2_file.minc2_to_numpy  # {MINC2_FLOAT: 'float32', ...}
minc2_file.numpy_to_minc2  # {'float32': MINC2_FLOAT, ...}

# Status
minc2_file.MINC2_SUCCESS   # 0
minc2_file.MINC2_ERROR     # -1
```

#### Exceptions

```python
minc2_error   # raised on any C-level error
```

---

### minc2_xfm

Read, write, and apply MINC transform files (`.xfm`).

```python
minc2_xfm(path=None)   # path: load file immediately
.open(path)
.save(path)
```

**Apply transforms**

```python
.transform_point(xyz)           # xyz: [x,y,z] or (n,3) array
.inverse_transform_point(xyz)   # apply inverse
```

**Inspect concatenated transforms**

```python
.get_n_concat()              # -> int: number of sub-transforms
.get_n_type(n=0)             # -> MINC2_XFM_LINEAR / MINC2_XFM_GRID_TRANSFORM / ...
.get_linear_transform(n=0)   # -> numpy.ndarray shape (4,4)
.get_linear_transform_param(n=0, center=None)  # -> minc2_transform_parameters
.get_grid_transform(n=0)     # -> (grid_file_path, inverted_bool)
```

**Build and modify transforms**

```python
.invert()
.append_linear_transform(matrix)     # matrix: (4,4) numpy.ndarray
.append_linear_param(params)          # params: minc2_transform_parameters
.append_grid_transform(grid_file, inv=False)
.concat_xfm(other)                    # concatenate another minc2_xfm
```

**minc2_transform_parameters** — parameter container

```python
p = minc2_transform_parameters()
p.center        # np.ndarray([3])
p.translations  # np.ndarray([3])
p.scales        # np.ndarray([3])
p.shears        # np.ndarray([3])
p.rotations     # np.ndarray([3])
p.invalid       # bool: True if extraction failed
```

**Transform type constants**

```python
minc2_xfm.MINC2_XFM_LINEAR
minc2_xfm.MINC2_XFM_THIN_PLATE_SPLINE
minc2_xfm.MINC2_XFM_CONCATENATED_TRANSFORM
minc2_xfm.MINC2_XFM_GRID_TRANSFORM
```

---

### minc2_tags

Read and write MINC tag files (`.tag`), which store point-in-volume landmarks.

```python
minc2_tags(path=None, n_volumes=1)
.load(path)
.save(path)

len(tags)           # number of tag points
tags.n_volumes      # 1 or 2
tags.tag            # list of np.ndarray, shape (n_tags, 3) per volume
tags.weights        # np.ndarray([n_tags]) or None
tags.structure_ids  # np.ndarray([n_tags], int) or None
tags.patient_ids    # np.ndarray([n_tags], int) or None
tags.labels         # list[str] or None
```

---

### minc2_input_iterator / minc2_output_iterator

Voxel-by-voxel iteration over one or more volumes. Useful for per-voxel
operations on large files without loading everything into memory.

```python
from minc2_simple import minc2_input_iterator, minc2_output_iterator

inp = minc2_input_iterator(files=['a.mnc', 'b.mnc', 'c.mnc'])
out = minc2_output_iterator(
    files=['result.mnc'],
    reference=inp,              # copy geometry from inp
    data_type=minc2_file.MINC2_FLOAT,
)

for values in inp:              # values: np.ndarray shape (n_files,)
    out.set_value([np.mean(values)])
    out.next()

inp.close()
out.close()
```

`reference` for the output iterator can be a `minc2_file`, a path string,
another iterator, or a list of `minc2_dim`.

---

## C API Reference

Include `minc2-simple.h`. All functions return `MINC2_SUCCESS` (0) or
`MINC2_ERROR` (-1) unless noted.

### Types

```c
typedef struct minc2_file*          minc2_file_handle;
typedef struct minc2_xfm_file*      minc2_xfm_file_handle;
typedef struct minc2_info_iterator* minc2_info_iterator_handle;
typedef struct minc2_file_iterator* minc2_file_iterator_handle;
typedef struct minc2_tags*          minc2_tags_handle;

struct minc2_dimension {
  int    id;           /* MINC2_DIM_X/Y/Z/TIME/VEC/UNKNOWN */
  int    length;
  int    irregular;
  double step;
  double start;
  int    have_dir_cos;
  double dir_cos[3];
};
```

### File lifecycle

```c
int minc2_allocate(minc2_file_handle *h);
minc2_file_handle minc2_allocate0(void);   /* allocate + init, returns NULL on error */
int minc2_init(minc2_file_handle h);
int minc2_free(minc2_file_handle h);
int minc2_destroy(minc2_file_handle h);    /* close if open, then free */

int minc2_open(minc2_file_handle h, const char *path);
int minc2_open_rdwr(minc2_file_handle h, const char *path);
int minc2_close(minc2_file_handle h);

int minc2_define(minc2_file_handle h,
                 struct minc2_dimension *store_dims,
                 int store_data_type,
                 int data_type);
int minc2_create(minc2_file_handle h, const char *path);
```

### Queries

```c
int minc2_ndim(minc2_file_handle h, int *ndim);
int minc2_nelement(minc2_file_handle h, int *nelement);
int minc2_data_type(minc2_file_handle h, int *type);
int minc2_storage_data_type(minc2_file_handle h, int *type);
int minc2_slice_ndim(minc2_file_handle h, int *slice_ndim);

int minc2_get_store_dimensions(minc2_file_handle h,
                               struct minc2_dimension **dims);
int minc2_get_representation_dimensions(minc2_file_handle h,
                                        struct minc2_dimension **dims);
int minc2_setup_standard_order(minc2_file_handle h);
```

### Data I/O

```c
/* Complete volume */
int minc2_load_complete_volume(minc2_file_handle h,
                               void *buffer,
                               int representation_type);
int minc2_save_complete_volume(minc2_file_handle h,
                               const void *buffer,
                               int representation_type);

/* Hyperslab */
int minc2_read_hyperslab(minc2_file_handle h,
                         int *start, int *count,
                         void *buffer,
                         int representation_type);
int minc2_write_hyperslab(minc2_file_handle h,
                          int *start, int *count,
                          const void *buffer,
                          int representation_type);
```

`start` and `count` are arrays of length `ndim` in representation order.

### Scaling

```c
int minc2_set_scaling(minc2_file_handle h,
                      int use_global_scaling,
                      int use_slice_scaling);
int minc2_set_volume_range(minc2_file_handle h,
                           double value_min,
                           double value_max);
int minc2_set_slice_range(minc2_file_handle h,
                          int *start,
                          double value_min,
                          double value_max);
```

### Coordinate transformations

```c
int minc2_voxel_to_world(minc2_file_handle h,
                         const double *voxel, double *world);
int minc2_world_to_voxel(minc2_file_handle h,
                         const double *world,  double *voxel);

/* Vectorized: n points, each point separated by stride doubles */
int minc2_voxel_to_world_vec(minc2_file_handle h, int n, int stride,
                             const double *voxel, double *world);
int minc2_world_to_voxel_vec(minc2_file_handle h, int n, int stride,
                             const double *world, double *voxel);
```

### Attributes and metadata

```c
int minc2_get_attribute_type(minc2_file_handle h,
                             const char *group, const char *attr,
                             int *minc2_type);
int minc2_get_attribute_length(minc2_file_handle h,
                               const char *group, const char *attr,
                               int *attr_length);
int minc2_read_attribute(minc2_file_handle h,
                         const char *group, const char *attr,
                         void *buf, int buf_size);
int minc2_write_attribute(minc2_file_handle h,
                          const char *group, const char *attr,
                          const void *buf, int buf_size,
                          int minc2_type);
int minc2_delete_attribute(minc2_file_handle h,
                           const char *group, const char *attr);
int minc2_delete_group(minc2_file_handle h, const char *group);
int minc2_copy_metadata(minc2_file_handle src, minc2_file_handle dst);
```

### Metadata iterators

```c
minc2_info_iterator_handle minc2_allocate_info_iterator(void);
int minc2_free_info_iterator(minc2_info_iterator_handle it);
int minc2_stop_info_iterator(minc2_info_iterator_handle it);

int minc2_start_group_iterator(minc2_file_handle h,
                               minc2_info_iterator_handle it);
int minc2_iterator_group_next(minc2_info_iterator_handle it);
const char *minc2_iterator_group_name(minc2_info_iterator_handle it);

int minc2_start_attribute_iterator(minc2_file_handle h,
                                   const char *group,
                                   minc2_info_iterator_handle it);
int minc2_iterator_attribute_next(minc2_info_iterator_handle it);
const char *minc2_iterator_attribute_name(minc2_info_iterator_handle it);
```

### Variable / dataset access

```c
int minc2_get_variable_ndims(minc2_file_handle h,
                             const char *path, const char *name,
                             int *ndims);
int minc2_get_variable_dims(minc2_file_handle h,
                            const char *path, const char *name,
                            int *dims);   /* pre-allocated, length ndims */
int minc2_get_variable_type(minc2_file_handle h,
                            const char *path, const char *name,
                            int *minc2_type);
int minc2_read_variable_raw(minc2_file_handle h,
                            const char *path, const char *name,
                            int representation_type,
                            int *start, int *count,
                            void *buffer);
int minc2_write_variable_raw(minc2_file_handle h,
                             const char *path, const char *name,
                             int representation_type,
                             int *start, int *count,
                             const void *buffer);
```

### Voxel iterator

```c
minc2_file_iterator_handle minc2_iterator_allocate0(void);
int minc2_iterator_free(minc2_file_iterator_handle h);

int minc2_iterator_input_start(minc2_file_iterator_handle h,
                               minc2_file_handle m, int data_type);
int minc2_iterator_output_start(minc2_file_iterator_handle h,
                                minc2_file_handle m, int data_type);
int minc2_multi_iterator_input_start(minc2_file_iterator_handle h,
                                     minc2_file_handle *m,
                                     int data_type, int fnum);
int minc2_multi_iterator_output_start(minc2_file_iterator_handle h,
                                      minc2_file_handle *m,
                                      int data_type, int fnum);

int minc2_iterator_next(minc2_file_iterator_handle h);
int minc2_iterator_get_values(minc2_file_iterator_handle h, void *val);
int minc2_iterator_put_values(minc2_file_iterator_handle h, const void *val);
```

### Transform (XFM) API

```c
minc2_xfm_file_handle minc2_xfm_allocate0(void);
int minc2_xfm_free(minc2_xfm_file_handle h);
int minc2_xfm_destroy(minc2_xfm_file_handle h);
int minc2_xfm_open(minc2_xfm_file_handle h, const char *path);
int minc2_xfm_save(minc2_xfm_file_handle h, const char *path);

int minc2_xfm_transform_point(minc2_xfm_file_handle h,
                              const double *in, double *out);
int minc2_xfm_inverse_transform_point(minc2_xfm_file_handle h,
                                      const double *in, double *out);
int minc2_xfm_transform_point_vec(minc2_xfm_file_handle h, int n, int stride,
                                  const double *in, double *out);
int minc2_xfm_inverse_transform_point_vec(minc2_xfm_file_handle h, int n, int stride,
                                          const double *in, double *out);

int minc2_xfm_get_n_concat(minc2_xfm_file_handle h, int *n);
int minc2_xfm_get_n_type(minc2_xfm_file_handle h, int n, int *xfm_type);
int minc2_xfm_get_linear_transform(minc2_xfm_file_handle h,
                                   int n, double *matrix); /* 4x4 row-major */
int minc2_xfm_get_grid_transform(minc2_xfm_file_handle h,
                                 int n, int *inverted, char **grid_file);

int minc2_xfm_invert(minc2_xfm_file_handle h);
int minc2_xfm_append_linear_transform(minc2_xfm_file_handle h, double *matrix);
int minc2_xfm_append_linear_param(minc2_xfm_file_handle h,
                                  double *center, double *translations,
                                  double *scales, double *shears,
                                  double *rotations);
int minc2_xfm_extract_linear_param(minc2_xfm_file_handle h, int n,
                                   double *center, double *translations,
                                   double *scales, double *shears,
                                   double *rotations);
int minc2_xfm_append_grid_transform(minc2_xfm_file_handle h,
                                    const char *grid_path, int inv);
int minc2_xfm_concat_xfm(minc2_xfm_file_handle h,
                         minc2_xfm_file_handle other);
```

### Tags API

```c
minc2_tags_handle minc2_tags_allocate0(void);
int minc2_tags_free(minc2_tags_handle tags);
int minc2_tags_load(minc2_tags_handle tags, const char *file);
int minc2_tags_save(minc2_tags_handle tags, const char *file);
int minc2_tags_init(minc2_tags_handle tags,
                    int n_tag_points, int n_volumes,
                    int have_weights, int have_structure_ids,
                    int have_patient_ids, int have_labels);
```

### Utility

```c
const char *minc2_data_type_name(int minc2_type_id);
const char *minc2_dim_type_name(int minc2_dim_id);
char *minc2_timestamp(int argc, char **argv); /* generate history string */
```

---

## Common Patterns

### Read, process, write (Python)

```python
from minc2_simple import minc2_file
import numpy as np

inp = minc2_file('input.mnc')
data = inp.data                 # np.ndarray, standard order (Z,Y,X or T,Z,Y,X)

result = data.astype(np.float64)
result[result < 0] = 0          # threshold negatives

out = minc2_file()
out.imitate(inp, store_type=minc2_file.MINC2_FLOAT)
out.create('output.mnc')
out.data = result
out.copy_metadata(inp)
out.close()
inp.close()
```

### Slice-by-slice (memory-efficient, Python)

```python
inp = minc2_file('big.mnc')
inp.setup_standard_order()

dims = inp.representation_dims()
nz = dims[0].length             # first dim after standard order

out = minc2_file()
out.imitate(inp)
out.create('out.mnc')
out.setup_standard_order()

for z in range(nz):
    sl = inp.vol[z, :, :]       # loads one slice
    out.vol[z, :, :] = sl * 2

inp.close()
out.close()
```

### Per-voxel average of N files (Python)

```python
from minc2_simple import minc2_input_iterator, minc2_output_iterator

files = ['a.mnc', 'b.mnc', 'c.mnc']
inp = minc2_input_iterator(files=files)
out = minc2_output_iterator(
    files=['avg.mnc'],
    reference=inp,
    data_type=minc2_file.MINC2_FLOAT,
)
for values in inp:
    out.set_value([float(np.mean(values))])
    out.next()
inp.close()
out.close()
```

### Apply a transform (Python)

```python
from minc2_simple import minc2_xfm
import numpy as np

xfm = minc2_xfm('subject_to_mni.xfm')

# Single point (world mm)
out_pt = xfm.transform_point(np.array([10.0, -20.0, 30.0]))

# Batch (n x 3)
pts = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
out_pts = xfm.transform_point(pts)

# Extract 4x4 linear matrix
mat = xfm.get_linear_transform(0)  # shape (4,4)
```

### Read, process, write (C)

```c
#include "minc2-simple.h"
#include <stdlib.h>

int main(void)
{
    minc2_file_handle in  = minc2_allocate0();
    minc2_file_handle out = minc2_allocate0();

    minc2_open(in, "input.mnc");
    minc2_setup_standard_order(in);

    int nelement;
    minc2_nelement(in, &nelement);

    double *data = malloc(nelement * sizeof(double));
    minc2_load_complete_volume(in, data, MINC2_DOUBLE);

    for (int i = 0; i < nelement; i++)
        data[i] *= 2.0;

    struct minc2_dimension *dims;
    minc2_get_store_dimensions(in, &dims);
    minc2_define(out, dims, MINC2_FLOAT, MINC2_DOUBLE);
    minc2_create(out, "output.mnc");
    minc2_setup_standard_order(out);
    minc2_save_complete_volume(out, data, MINC2_DOUBLE);
    minc2_copy_metadata(in, out);

    minc2_close(in);
    minc2_close(out);
    minc2_free(in);
    minc2_free(out);
    free(data);
    return 0;
}
```

### Enumerate metadata groups and attributes (C)

```c
minc2_info_iterator_handle git = minc2_allocate_info_iterator();
minc2_info_iterator_handle ait = minc2_allocate_info_iterator();

minc2_start_group_iterator(h, git);
while (minc2_iterator_group_next(git) == MINC2_SUCCESS) {
    const char *grp = minc2_iterator_group_name(git);
    minc2_start_attribute_iterator(h, grp, ait);
    while (minc2_iterator_attribute_next(ait) == MINC2_SUCCESS)
        printf("%s / %s\n", grp, minc2_iterator_attribute_name(ait));
    minc2_stop_info_iterator(ait);
}
minc2_free_info_iterator(ait);
minc2_free_info_iterator(git);
```

---

## Key Concepts

### Dimension order and standard order

MINC2 files store dimensions in an arbitrary order dictated by the file.
`setup_standard_order()` (C) / `.setup_standard_order()` (Python)
reorders the in-memory view to:

```
slowest  TIME → Z → Y → X → VEC  fastest
```

It also flips any axis with a negative step so all steps are positive
(start is adjusted accordingly). This makes data arrays compatible with
numpy C-order (row-major) conventions. Always call this after `open()` or
`create()` before doing I/O unless you specifically need file order.

### Storage vs. representation types

- **Storage type** — how data is saved on disk (e.g., `MINC2_SHORT` for 16-bit MRI).
- **Representation type** — the format requested for the in-memory buffer.

When loading, the library scales stored integers to the requested type using
per-volume or per-slice min/max attributes. When you pass `MINC2_FLOAT` or
`MINC2_DOUBLE` as the representation type to `load_complete_volume()`, you
get scaled floating-point data regardless of the on-disk format.

### Scaling modes

| Mode | API | Behaviour |
|------|-----|-----------|
| None | default | raw integer values, no scaling |
| Global | `set_scaling(1, 0)` + `set_volume_range()` | one scale factor for whole file |
| Slice | `set_scaling(0, 1)` + `set_slice_range()` | separate scale per slice |

For Python, pass `global_scaling=True` or `slice_scaling=True` to `define()`.

### Error handling

**C:** every function returns `MINC2_SUCCESS` (0) or `MINC2_ERROR` (-1). Check every call.

**Python:** errors raise `minc2_error`. The library does not use return codes.

### Resource management

**C:** pair every `minc2_allocate*` with `minc2_free` or `minc2_destroy`.
Call `minc2_close` before reading a file you just wrote.

**Python:** files are closed on garbage collection via `ffi.gc()`, but call
`.close()` explicitly to ensure timely flushing and to allow re-reading
immediately after writing.

---

## Building

### minc2-simple (C library)

```bash
cd subproject/minc2-simple
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLIBMINC_DIR=/path/to/libminc/build-or-install
cmake --build build -j$(nproc)
cd build && ctest --verbose
```

### Python bindings

The CFFI build script needs a `MINC_TOOLKIT` directory with `include/` and
`lib/` subdirectories. The simplest setup is to build and install libminc as
a shared library:

```bash
# 1. Build and install libminc as shared
cd subproject/libminc
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLIBMINC_BUILD_SHARED_LIBS=ON \
  -DLIBMINC_MINC1_SUPPORT=ON \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install
cmake --build build -j$(nproc)
cmake --install build

# 2. Install Python deps
pip3 install cffi numpy setuptools

# 3. Build the CFFI extension
cd subproject/minc2-simple/python
MINC_TOOLKIT=/path/to/libminc/install \
  python3 setup.py build_ext --inplace

# 4. Run tests
python3 -m unittest test.minc2_variable -v      # variable API (no external tools needed)
python3 -m unittest discover -s test -v         # all tests (needs rawtominc on PATH)
```

See `plan.md` for detailed instructions and alternative (static linking) approaches.
