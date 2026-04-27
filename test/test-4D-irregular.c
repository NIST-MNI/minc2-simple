/*
 * test-4D-irregular: end-to-end test for irregular time-dimension support
 * in the minc2-simple C facade.
 *
 * Cycles (TDD):
 *   1. Open fixture → time dim has irregular flag set.
 *   2. Open fixture → time dim's per-sample `offsets` array is populated
 *      and matches the known per-sample times.
 *   3. Define + create a new file with synthesised offsets, reopen, assert
 *      offsets round-trip.
 *   4. minc2_compare_dimensions distinguishes two irregular dims that
 *      differ only in their offsets array.
 *
 * Argv:
 *   argv[1] = path to test_4D_irregular_offsets.mnc fixture (read).
 *   argv[2] = path to scratch .mnc the test writes + re-reads (write).
 */

#include "minc2-simple.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXP_LEN 6
static const double EXPECTED_OFFSETS[EXP_LEN] =
    { 0.0, 120.0, 300.0, 800.0, 1200.0, 1500.0 };
static const double EXPECTED_WIDTHS[EXP_LEN] =
    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

#define FAIL_IF(cond, msg)                                                 \
    do {                                                                   \
        if (cond) {                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);  \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int find_time(struct minc2_dimension *dims)
{
    int i;
    for (i = 0; dims[i].id != MINC2_DIM_END; i++)
        if (dims[i].id == MINC2_DIM_TIME) return i;
    return -1;
}

/* Cycles 1 + 2: read existing fixture, verify irregular flag and offsets. */
static int test_read(const char *path)
{
    minc2_file_handle h = minc2_allocate0();
    struct minc2_dimension *dims;
    int t, i;

    FAIL_IF(minc2_open(h, path) != MINC2_SUCCESS, "open fixture");
    FAIL_IF(minc2_get_representation_dimensions(h, &dims) != MINC2_SUCCESS,
            "get dims");

    t = find_time(dims);
    FAIL_IF(t < 0, "fixture must contain a time dimension");

    /* Cycle 1 */
    FAIL_IF(dims[t].irregular != 1, "time dim should be irregular");
    FAIL_IF(dims[t].length != EXP_LEN, "time length");

    /* Cycle 2 */
    FAIL_IF(dims[t].offsets == NULL, "time offsets should be populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(dims[t].offsets[i] - EXPECTED_OFFSETS[i]) > 1e-9) {
            fprintf(stderr,
                    "FAIL offsets[%d]: expected %g, got %g\n",
                    i, EXPECTED_OFFSETS[i], dims[t].offsets[i]);
            return 1;
        }
    }

    /* Cycle 5: widths should be populated for irregular time dim */
    FAIL_IF(dims[t].widths == NULL, "time widths should be populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(dims[t].widths[i] - EXPECTED_WIDTHS[i]) > 1e-9) {
            fprintf(stderr,
                    "FAIL widths[%d]: expected %g, got %g\n",
                    i, EXPECTED_WIDTHS[i], dims[t].widths[i]);
            return 1;
        }
    }

    FAIL_IF(minc2_close(h) != MINC2_SUCCESS, "close fixture");
    minc2_free(h);
    return 0;
}

/* Cycle 3: define+create with synthesised offsets, reopen, verify round-trip. */
static int test_write(const char *out_path)
{
    minc2_file_handle h_out, h_in;
    struct minc2_dimension dims[5];
    struct minc2_dimension *got;
    /* Distinct from the fixture so we don't accidentally read stale state */
    double my_offsets[EXP_LEN] = { 0.5, 1.5, 4.0, 9.0, 16.0, 25.0 };
    float buf[EXP_LEN * 5 * 5 * 5];
    int i, t, n = EXP_LEN * 5 * 5 * 5;

    memset(dims, 0, sizeof(dims));
    dims[0].id = MINC2_DIM_TIME; dims[0].length = EXP_LEN;
    dims[0].irregular = 1; dims[0].offsets = my_offsets;
    dims[0].step = 1.0; dims[0].start = 0.0;
    dims[1].id = MINC2_DIM_Z; dims[1].length = 5;
    dims[1].step = 1.0; dims[1].start = 0.0;
    dims[2].id = MINC2_DIM_Y; dims[2].length = 5;
    dims[2].step = 1.0; dims[2].start = 0.0;
    dims[3].id = MINC2_DIM_X; dims[3].length = 5;
    dims[3].step = 1.0; dims[3].start = 0.0;
    dims[4].id = MINC2_DIM_END;

    h_out = minc2_allocate0();
    FAIL_IF(minc2_define(h_out, dims, MINC2_FLOAT, MINC2_FLOAT) != MINC2_SUCCESS,
            "define out");
    FAIL_IF(minc2_create(h_out, out_path) != MINC2_SUCCESS, "create out");

    for (i = 0; i < n; i++) buf[i] = (float)i;
    FAIL_IF(minc2_save_complete_volume(h_out, buf, MINC2_FLOAT) != MINC2_SUCCESS,
            "save volume");
    FAIL_IF(minc2_close(h_out) != MINC2_SUCCESS, "close out");
    minc2_free(h_out);

    h_in = minc2_allocate0();
    FAIL_IF(minc2_open(h_in, out_path) != MINC2_SUCCESS, "reopen out");
    FAIL_IF(minc2_get_representation_dimensions(h_in, &got) != MINC2_SUCCESS,
            "get reopened dims");

    t = find_time(got);
    FAIL_IF(t < 0, "reopened file must contain a time dimension");
    FAIL_IF(got[t].irregular != 1, "reopened time should be irregular");
    FAIL_IF(got[t].offsets == NULL, "reopened offsets should be populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(got[t].offsets[i] - my_offsets[i]) > 1e-9) {
            fprintf(stderr,
                    "FAIL roundtrip offsets[%d]: expected %g, got %g\n",
                    i, my_offsets[i], got[t].offsets[i]);
            return 1;
        }
    }

    /* Cycle 6: when the caller doesn't supply widths, libminc's
       miget_dimension_widths falls back to fabs(step). The dim above is
       defined with step=1.0, so we expect 1.0 widths. (See
       libminc/libsrc2/dimension.c:1577–1591 for the fallback chain.) */
    FAIL_IF(got[t].widths == NULL, "reopened widths should be populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(got[t].widths[i] - 1.0) > 1e-9) {
            fprintf(stderr,
                    "FAIL default widths[%d]: expected 1.0 (=fabs(step)), got %g\n",
                    i, got[t].widths[i]);
            return 1;
        }
    }

    FAIL_IF(minc2_close(h_in) != MINC2_SUCCESS, "close reopened");
    minc2_free(h_in);
    return 0;
}

/* Cycle 4: comparison must distinguish two irregular dims with different
 * offsets arrays. We synthesise two trivial dim arrays in memory. */
static int test_compare(void)
{
    double off_a[EXP_LEN] = { 0.0, 120.0, 300.0, 800.0, 1200.0, 1500.0 };
    double off_b[EXP_LEN] = { 0.0, 120.0, 300.0, 801.0, 1200.0, 1500.0 };
    struct minc2_dimension a[2], b[2];
    int rc_eq, rc_neq;

    memset(a, 0, sizeof(a)); memset(b, 0, sizeof(b));
    a[0].id = b[0].id = MINC2_DIM_TIME;
    a[0].length = b[0].length = EXP_LEN;
    a[0].irregular = b[0].irregular = 1;
    a[0].step = b[0].step = 1.0;
    a[0].start = b[0].start = 0.0;
    a[0].offsets = off_a;
    b[0].offsets = off_a;        /* same pointer → equal */
    a[1].id = b[1].id = MINC2_DIM_END;

    rc_eq = minc2_compare_dimensions(a, b);
    FAIL_IF(rc_eq != MINC2_SUCCESS, "identical irregular dims should compare equal");

    b[0].offsets = off_b;        /* different array → not equal */
    rc_neq = minc2_compare_dimensions(a, b);
    FAIL_IF(rc_neq == MINC2_SUCCESS,
            "irregular dims with different offsets must compare unequal");

    /* Cycle 7: widths comparison */
    double wid_a[EXP_LEN] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    double wid_b[EXP_LEN] = { 2.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    a[0].offsets = off_a;
    b[0].offsets = off_a;
    a[0].widths = wid_a;
    b[0].widths = wid_a;       /* same pointer → equal */

    rc_eq = minc2_compare_dimensions(a, b);
    FAIL_IF(rc_eq != MINC2_SUCCESS, "identical irregular dims with widths should compare equal");

    b[0].widths = wid_b;       /* different array → not equal */
    rc_neq = minc2_compare_dimensions(a, b);
    FAIL_IF(rc_neq == MINC2_SUCCESS,
            "irregular dims with different widths must compare unequal");

    return 0;
}

/* Cycle 8: full data + metadata round-trip. Reads the fixture's dims
 * AND voxel data, defines a new file using those exact dims, writes
 * the same data through, reopens, and asserts both the offsets and
 * the voxel buffer survive the round-trip byte-for-byte. */
static int test_full_cycle(const char *fixture_path, const char *out_path)
{
    minc2_file_handle h_src, h_out, h_back;
    struct minc2_dimension *src_store, *back_store;
    float *src_buf = NULL, *back_buf = NULL;
    int n_vox, store_type, t_back, i;

    /* 1. Read source: dims + data + storage type. */
    h_src = minc2_allocate0();
    FAIL_IF(minc2_open(h_src, fixture_path) != MINC2_SUCCESS, "fc: open src");
    FAIL_IF(minc2_get_store_dimensions(h_src, &src_store) != MINC2_SUCCESS,
            "fc: get src dims");
    FAIL_IF(minc2_nelement(h_src, &n_vox) != MINC2_SUCCESS, "fc: nelement");
    FAIL_IF(minc2_storage_data_type(h_src, &store_type) != MINC2_SUCCESS,
            "fc: storage dtype");

    src_buf = (float*)malloc((size_t)n_vox * sizeof(float));
    FAIL_IF(!src_buf, "fc: malloc src_buf");
    FAIL_IF(minc2_load_complete_volume(h_src, src_buf, MINC2_FLOAT)
            != MINC2_SUCCESS, "fc: load src");

    /* 2. Define + create + write a new file using the source's dims.
       minc2_define deep-copies the offsets, so we can free h_src after. */
    h_out = minc2_allocate0();
    FAIL_IF(minc2_define(h_out, src_store, store_type, MINC2_FLOAT)
            != MINC2_SUCCESS, "fc: define out");
    FAIL_IF(minc2_create(h_out, out_path) != MINC2_SUCCESS, "fc: create out");
    FAIL_IF(minc2_save_complete_volume(h_out, src_buf, MINC2_FLOAT)
            != MINC2_SUCCESS, "fc: save out");
    FAIL_IF(minc2_close(h_out) != MINC2_SUCCESS, "fc: close out");
    minc2_free(h_out);

    /* h_src no longer needed; src_buf stays alive for the comparison. */
    minc2_close(h_src);
    minc2_free(h_src);

    /* 3. Reopen new file, read dims + data, compare against source. */
    h_back = minc2_allocate0();
    FAIL_IF(minc2_open(h_back, out_path) != MINC2_SUCCESS, "fc: open back");
    FAIL_IF(minc2_get_store_dimensions(h_back, &back_store) != MINC2_SUCCESS,
            "fc: get back dims");

    t_back = find_time(back_store);
    FAIL_IF(t_back < 0, "fc: time dim found in back");
    FAIL_IF(back_store[t_back].irregular != 1, "fc: back time still irregular");
    FAIL_IF(back_store[t_back].length != EXP_LEN, "fc: back time length");
    FAIL_IF(back_store[t_back].offsets == NULL, "fc: back offsets populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(back_store[t_back].offsets[i] - EXPECTED_OFFSETS[i]) > 1e-9) {
            fprintf(stderr,
                    "FAIL full-cycle offsets[%d]: expected %g, got %g\n",
                    i, EXPECTED_OFFSETS[i], back_store[t_back].offsets[i]);
            free(src_buf);
            return 1;
        }
    }

    /* Cycle 9: widths survive full round-trip */
    FAIL_IF(back_store[t_back].widths == NULL, "fc: back widths populated");
    for (i = 0; i < EXP_LEN; i++) {
        if (fabs(back_store[t_back].widths[i] - EXPECTED_WIDTHS[i]) > 1e-9) {
            fprintf(stderr,
                    "FAIL full-cycle widths[%d]: expected %g, got %g\n",
                    i, EXPECTED_WIDTHS[i], back_store[t_back].widths[i]);
            free(src_buf);
            return 1;
        }
    }

    back_buf = (float*)malloc((size_t)n_vox * sizeof(float));
    FAIL_IF(!back_buf, "fc: malloc back_buf");
    FAIL_IF(minc2_load_complete_volume(h_back, back_buf, MINC2_FLOAT)
            != MINC2_SUCCESS, "fc: load back");

    for (i = 0; i < n_vox; i++) {
        if (fabs((double)back_buf[i] - (double)src_buf[i]) > 1e-5) {
            fprintf(stderr,
                    "FAIL full-cycle data[%d]: src=%g back=%g\n",
                    i, (double)src_buf[i], (double)back_buf[i]);
            free(src_buf);
            free(back_buf);
            return 1;
        }
    }

    free(src_buf);
    free(back_buf);
    minc2_close(h_back);
    minc2_free(h_back);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fixture.mnc> <scratch.mnc>\n", argv[0]);
        return 2;
    }
    if (test_read(argv[1])                    != 0) return 1;
    if (test_write(argv[2])                   != 0) return 1;
    if (test_compare()                        != 0) return 1;
    if (test_full_cycle(argv[1], argv[2])     != 0) return 1;
    fprintf(stdout, "test-4D-irregular: all cycles passed\n");
    return 0;
}
