/*
 * Generator for test_4D_irregular_offsets.mnc.
 *
 * Builds a 4D MINC2 volume whose time dimension is *irregularly* sampled,
 * with per-sample world coordinates (offsets) of:
 *     0, 120, 300, 800, 1200, 1500  (units: seconds)
 *
 * The other three dimensions (z, y, x) are 5-long, regularly sampled,
 * step = 1.0, start = 0.0. Voxel values are deterministic:
 *     voxel(t, z, y, x) = t*1000 + z*100 + y*10 + x   (as float)
 * so the test code can verify both the offsets and the data round-trip.
 *
 * Built but not registered as a CTest. Run once to (re-)produce the
 * committed fixture file.
 *
 * Usage:  gen-test_4D_irregular_offsets <output.mnc>
 */

#include <stdio.h>
#include <stdlib.h>
#include "minc2.h"

#define NDIMS 4
#define CT 6
#define CZ 5
#define CY 5
#define CX 5

int main(int argc, char **argv)
{
    midimhandle_t hdim[NDIMS];
    mihandle_t hvol;
    int r;
    misize_t i, t, z, y, x;
    misize_t start[NDIMS], count[NDIMS];
    double offsets[CT] = { 0.0, 120.0, 300.0, 800.0, 1200.0, 1500.0 };
    float *buf;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output.mnc>\n", argv[0]);
        return 1;
    }

    /* Order: slowest-varying first (time, z, y, x). */
    r = micreate_dimension("time",   MI_DIMCLASS_TIME,
                           MI_DIMATTR_NOT_REGULARLY_SAMPLED, CT, &hdim[0]);
    if (r < 0) { fprintf(stderr, "micreate_dimension(time) failed\n"); return 2; }

    r = micreate_dimension("zspace", MI_DIMCLASS_SPATIAL,
                           MI_DIMATTR_REGULARLY_SAMPLED, CZ, &hdim[1]);
    if (r < 0) { fprintf(stderr, "micreate_dimension(zspace) failed\n"); return 2; }

    r = micreate_dimension("yspace", MI_DIMCLASS_SPATIAL,
                           MI_DIMATTR_REGULARLY_SAMPLED, CY, &hdim[2]);
    if (r < 0) { fprintf(stderr, "micreate_dimension(yspace) failed\n"); return 2; }

    r = micreate_dimension("xspace", MI_DIMCLASS_SPATIAL,
                           MI_DIMATTR_REGULARLY_SAMPLED, CX, &hdim[3]);
    if (r < 0) { fprintf(stderr, "micreate_dimension(xspace) failed\n"); return 2; }

    /* Per-sample times for the irregular time dimension. */
    r = miset_dimension_offsets(hdim[0], CT, 0, offsets);
    if (r < 0) { fprintf(stderr, "miset_dimension_offsets failed\n"); return 3; }

    /* Regular spatial dims: unit step, zero start. */
    miset_dimension_separation(hdim[1], 1.0);
    miset_dimension_separation(hdim[2], 1.0);
    miset_dimension_separation(hdim[3], 1.0);
    miset_dimension_start(hdim[1], 0.0);
    miset_dimension_start(hdim[2], 0.0);
    miset_dimension_start(hdim[3], 0.0);

    r = micreate_volume(argv[1], NDIMS, hdim, MI_TYPE_FLOAT, MI_CLASS_REAL,
                        NULL, &hvol);
    if (r < 0) { fprintf(stderr, "micreate_volume failed\n"); return 4; }

    r = micreate_volume_image(hvol);
    if (r < 0) { fprintf(stderr, "micreate_volume_image failed\n"); return 4; }

    buf = (float *) malloc(CT * CZ * CY * CX * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 5; }
    i = 0;
    for (t = 0; t < CT; t++)
        for (z = 0; z < CZ; z++)
            for (y = 0; y < CY; y++)
                for (x = 0; x < CX; x++)
                    buf[i++] = (float)(t * 1000 + z * 100 + y * 10 + x);

    start[0] = start[1] = start[2] = start[3] = 0;
    count[0] = CT; count[1] = CZ; count[2] = CY; count[3] = CX;
    r = miset_real_value_hyperslab(hvol, MI_TYPE_FLOAT, start, count, buf);
    if (r < 0) { fprintf(stderr, "miset_real_value_hyperslab failed\n"); free(buf); return 6; }

    free(buf);
    r = miclose_volume(hvol);
    if (r < 0) { fprintf(stderr, "miclose_volume failed\n"); return 7; }

    fprintf(stdout, "Wrote %s (4D irregular: time=%d offsets, %dx%dx%d spatial)\n",
            argv[1], CT, CZ, CY, CX);
    return 0;
}
