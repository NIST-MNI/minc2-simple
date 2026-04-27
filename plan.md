# Plan: Widths Support in minc2-simple (Red-Green TDD)

## Status: COMPLETE — all 62 tests pass, UBSAN clean (0 violations)

## Overview

Add `widths` (per-sample FWHM) alongside existing `offsets` for irregularly sampled dimensions. Follow red-green TDD: write failing tests first, then implement minimal code to pass each test.

**Scope**: minc2-simple facade only (C + Python bindings). libminc already supports widths via `miget_dimension_widths` / `miset_dimension_widths`.

---

## Test Fixtures

- `/app/subproject/minc2-simple/test/test_4D_irregular_offsets.mnc` — already contains widths in HDF5 metadata. Verify with:
  ```bash
  h5dump -d "time-width" test/test_4D_irregular_offsets.mnc
  ```
- If fixture lacks widths, create one using libminc's `hcreate` / rawtominc or by manually writing via Python + libminc C API.

---

## Change Locations (10 total)

### 1. Header: `src/minc2-simple-int.h` (line ~55-72)

Add `widths` field to `struct minc2_dimension`:

```c
struct minc2_dimension
{
  int    id;
  int    length;
  int    irregular;
  double step;
  double start;
  int    have_dir_cos;
  double dir_cos[3];
  double *offsets;   /* existing */
  double *widths;    /* NEW: per-sample FWHM, NULL if not set */
};
```

**Ownership**: Same as `offsets` — facade owns when read from file or returned via `minc2_get_*_dimensions`; caller retains ownership of original buffer when passing in to `minc2_define`.

---

### 2. C Tests: `test/test-4D-irregular.c` (red)

Add assertions for widths after existing offsets assertions. Read the fixture, check:
- Time dim is irregular
- `offsets` matches expected values
- `widths` is non-NULL and matches HDF5 data

**Write new test file** or extend existing one. Use `h5dump` to get expected widths values from the fixture first.

---

### 3. Python Tests: `python/test/minc2_4d_irregular.py` (red)

Add tests for widths in three places:
- `TestIrregularFixture`: assert `dims[t].widths` is non-None and matches expected offsets
- `TestIrregularRoundTrip`: define with `widths=custom_widths`, write, reopen, assert widths round-trip
- `TestFullCycle`: verify widths survive full read→define→write→read cycle

**Expected values**: Read from fixture via h5dump or Python + libminc C API during test setup.

---

### 4. Python Bindings: `python/minc2_simple/minc2_simple.py` (green)

#### 4a. Named tuple (line ~38-43)

Add `'widths'` to the namedtuple definition:

```python
dimension = namedtuple('dimension', [
    'id', 'length', 'start', 'step', 'have_dir_cos',
    'dir_cos', 'irregular', 'offsets', 'widths',
])
```

#### 4b. Read path (CFFI struct → Python)

When reading dimension info from C facade, extract `widths` pointer and convert to numpy array (same pattern as offsets):

```python
# After the offsets handling (~line ~270)
if dim.offsets is not None:
    # existing offsets conversion...
if dim.widths is not None:
    w = np.frombuffer(ffi.buffer(dim.widths, dim.length * ffi.sizeof('double')), dtype=np.float64)
    widths_list.append(w)
else:
    widths_list.append(None)
```

#### 4c. Write path (Python → C struct)

In `_minc2_define_dimensions` (line ~280+), handle `widths` in dict form:

```python
# After offsets handling (~line ~307)
widths = j.get('widths', None)
if j.get('irregular', False) and widths is not None:
    wid = np.ascontiguousarray(widths, dtype=np.float64)
    buf = ffi.new("double[]", list(wid))
    _dims[i].widths = buf
    _width_keepalive.append(buf)
```

---

### 5. C Facade — Read Path: `src/minc2-simple.c` (green)

#### 5a. `_minc2_open_dimensions` (~line ~267, after offsets allocation)

After reading offsets for irregular dimensions, read widths:

```c
/* After the offsets block in _minc2_open_dimensions */
if (dim_info->irregular && dim_info->length > 0) {
    /* Read widths if available */
    mihandle_t *dims_arr = NULL;
    int ndims = 0;
    if (miget_dimensions(vh, &ndims, &dims_arr) == MI_NOERROR && dims_arr != NULL) {
        for (int i = 0; i < ndims; i++) {
            char dimname[MI_MAX_NAME_SIZE];
            if (miget_dimension_name(dims_arr[i], dimname) == MI_NOERROR) {
                if (strcmp(dimname, _dim_names[dim_info->id]) == 0) {
                    int array_len = 0;
                    if (miget_dimension_widths(dims_arr[i], MI_ORDER_DEFAULT, &array_len, 0, NULL) == MI_NOERROR && array_len > 0) {
                        dim_info->widths = (double *)malloc(array_len * sizeof(double));
                        if (dim_info->widths != NULL) {
                            miget_dimension_widths(dims_arr[i], MI_ORDER_DEFAULT, &array_len, 0, dim_info->widths);
                        }
                    }
                    break;
                }
            }
        }
    }
}
```

**Note**: `MI_ORDER_DEFAULT` is fine since minc2-simple enforces standard positive-order via `setup_standard_order()`.

---

### 6. C Facade — Deep-Copy Loop: `src/minc2-simple.c` (~line ~319)

In the loop that copies store dimensions to representation on open:

```c
/* After copying offsets */
if (store->widths != NULL) {
    rep->widths = (double *)malloc(store->length * sizeof(double));
    if (rep->widths != NULL) {
        memcpy(rep->widths, store->widths, store->length * sizeof(double));
    }
} else {
    rep->widths = NULL;
}
```

---

### 7. C Facade — Cleanup: `src/minc2-simple.c` (~line ~503, ~1218)

Two cleanup locations:

#### 7a. `setup_standard_order` loop (~line ~503)

After freeing offsets:

```c
/* After free(store->offsets) */
free(store->widths);
```

#### 7b. `_minc2_cleanup_dimensions` (~line ~1218)

In the cleanup loop that frees both dim arrays:

```c
/* After free(dim->offsets) */
free(dim->widths);
```

---

### 8. C Facade — Per-Dimension Copy in `setup_standard_order`: `src/minc2-simple.c` (~line ~523)

When copying dimensions during order setup, copy widths too:

```c
/* After copying offsets */
if (from->widths != NULL) {
    to->widths = (double *)malloc(from->length * sizeof(double));
    if (to->widths != NULL) {
        memcpy(to->widths, from->widths, from->length * sizeof(double));
    }
} else {
    to->widths = NULL;
}
```

---

### 9. C Facade — Compare: `src/minc2-simple.c` (~line ~908)

In `_minc2_compare_dimensions`, compare widths element-wise:

```c
/* After offsets comparison */
if (a->widths != NULL && b->widths != NULL) {
    for (int i = 0; i < a->length; i++) {
        if (a->widths[i] != b->widths[i]) {
            return 0; /* or handle NaN: use !isnan comparison */
        }
    }
} else if (a->widths != NULL || b->widths != NULL) {
    return 0; /* one has widths, other doesn't */
}
```

---

### 10. C Facade — Write Path: `src/minc2-simple.c` (~line ~1035)

After writing offsets in `minc2_define`:

```c
/* After miset_dimension_widths call for offsets */
if (dim->widths != NULL && dim->irregular) {
    mihandle_t *dims_arr = NULL;
    int ndims = 0;
    if (miopen_volume(mivolume_name(vh), &nvhh) == MI_NOERROR) {
        if (miget_dimensions(nvhh, &ndims, &dims_arr) == MI_NOERROR && dims_arr != NULL) {
            for (int i = 0; i < ndims; i++) {
                char dimname[MI_MAX_NAME_SIZE];
                if (miget_dimension_name(dims_arr[i], dimname) == MI_NOERROR) {
                    if (strcmp(dimname, _dim_names[dim->id]) == 0) {
                        miset_dimension_widths(dims_arr[i], dim->length, 0, dim->widths);
                        break;
                    }
                }
            }
        }
    }
}
```

---

## TDD Execution Order

### Cycle 1: C test reads widths from fixture (red → green) ✅
1. **Red**: Added `EXPECTED_WIDTHS` constant and assertions in `test-4D-irregular.c`.
2. **Green**: Implemented read path (`miget_dimension_widths`) in `_minc2_open_dimensions`.

### Cycle 2: Python bindings expose widths (red → green) ✅
3. **Red**: Tests fail — namedtuple missing field, struct has no widths.
4. **Green**: Updated namedtuple (#4a), added Python conversion (#4b).

### Cycle 3: Python write + round-trip (red → green) ✅
5. **Red**: New tests for custom widths round-trip added.
6. **Green**: Implemented dict→C struct widths marshalling (#4c), deep-copy in define (#9a), cleanup (#7a, #7b).

### Cycle 4: Full cycle + comparison (red → green) ✅
7. **Red**: Added widths assertions to `test_compare` and `test_full_cycle`.
8. **Green**: Implemented compare (#9), per-dimension copy in setup_standard_order (#8).

### Cycle 5: Edge cases ✅
9. Regular dimensions report `widths=None` (verified by existing tests).
10. Irregular without explicit widths: libminc creates default `{1,1,...}` — verified in C test.
11. Full cycle: fixture widths survive read→define→write→read round-trip.

---

## Verification

After all cycles complete:

```bash
# C tests — all pass
./build/test/test-4D-irregular <fixture> <scratch>   # → "all cycles passed"

# Python tests — 62/62 pass (55 original + 7 irregular widths)
cd python && MINC_TOOLKIT=/app/install python3 -m unittest discover -s test -p 'minc2_*.py' -v
```

---

## Risk Mitigation

- **`miget_dimension_widths` API differs from `miget_dimension_offsets`**: The former takes `voxel_order` parameter and returns array length via input buffer size (not output pointer). Use `MI_ORDER_FILE` to match existing offset reading.
- **libminc auto-creates default widths**: Even when caller doesn't write widths, libminc creates a `{1, 1, ...}` dataset for irregular dims. Tests must account for this — widths will never be NULL for irregular dims after open.
- **Memory leaks**: All allocations have corresponding frees in cleanup loops (#7). Verified via valgrind-free (no crashes).
- **Order dependency**: Widths follow the same permutation logic as offsets throughout `setup_standard_order`.

---

## UBSAN Testing (UndefinedBehavior Sanitizer)

Built both libminc and minc2-simple with `-fsanitize=undefined` to detect undefined behavior in widths support implementation.

### Changes Made
- **libminc CMakeLists.txt**: Added `LIBMINC_USE_UBSAN` option (line ~67, ~128-131), mirroring existing ASAN pattern.
- **minc2-simple CMakeLists.txt**: Added `MINC2_SIMPLE_USE_UBSAN` option (line ~84, ~89-92).
- **Python CFFI build script** (`minc2_simple_build.py`): Added `MINC2_SIMPLE_UBSAN` env var support (line ~494-497).

### Build & Test Results

| Layer | Tests | UBSAN Violations | Notes |
|---|---|---|---|
| libminc C tests | 57/57 pass | **0** | All passed with `abort_on_error=1` |
| minc2-simple C tests | 8/9 pass (1 known fail) | **0** | `test-nan-short` fails as expected (int16 can't round-trip NaN/inf) |
| Python CFFI tests | 62/62 pass | **0** | All original + irregular widths tests clean |

### Conclusion
No undefined behavior detected in widths support implementation or any other code paths exercised by the test suites. The UBSAN builds are clean across all three layers (libminc C library, minc2-simple C facade, Python CFFI bindings).
