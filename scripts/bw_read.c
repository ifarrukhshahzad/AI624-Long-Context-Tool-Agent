/*
 * bw_read.c - Pure READ memory bandwidth test (what LLM decode actually does).
 *
 * Why this exists: STREAM counts only the bytes the program asks for. On a
 * write, the CPU first READS the destination line into cache ("write-allocate"),
 * so STREAM Copy/Triad move more real DRAM traffic than they report. LLM decode
 * is almost pure reading of weights + KV cache, so we measure pure reads.
 *
 * Method: fill a large array (far bigger than the 2 MB L3 cache), then sum it
 * with all threads, several times; report the best run in GB/s (1 GB = 1e9 B).
 *
 * Build and run on the Pi:
 *   gcc -O3 -fopenmp scripts/bw_read.c -o bw_read
 *   OMP_NUM_THREADS=4 ./bw_read            (default 1 GiB array, 10 reps)
 *   OMP_NUM_THREADS=4 ./bw_read 2048 10    (2 GiB array, 10 reps)
 */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t mib = (argc > 1) ? (size_t)atol(argv[1]) : 1024;
    int reps = (argc > 2) ? atoi(argv[2]) : 10;
    size_t n = mib * 1024 * 1024 / sizeof(uint64_t);

    uint64_t *a = (uint64_t *)aligned_alloc(64, n * sizeof(uint64_t));
    if (!a) { fprintf(stderr, "allocation failed\n"); return 1; }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = i;  /* first touch, all pages real */

    double best = 1e30;
    uint64_t check = 0;
    for (int r = 0; r < reps; r++) {
        uint64_t s = 0;
        double t0 = omp_get_wtime();
        #pragma omp parallel for reduction(+ : s) schedule(static)
        for (size_t i = 0; i < n; i++) s += a[i];
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
        check += s;
    }
    double bytes = (double)n * sizeof(uint64_t);
    printf("Threads      : %d\n", omp_get_max_threads());
    printf("Array        : %zu MiB\n", mib);
    printf("Best time    : %.4f s\n", best);
    printf("Read BW      : %.2f GB/s (1 GB = 1e9 bytes)\n", bytes / best / 1e9);
    printf("(checksum %llu, ignore)\n", (unsigned long long)check);
    free(a);
    return 0;
}
