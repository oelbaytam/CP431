#include <stddef.h>
#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


typedef unsigned long long ull;

// Sieve odd primes in [3..limit]. 2 handled separately.
static uint32_t* simple_sieve_odd(uint32_t limit, int* count_out) {
    uint8_t* is_comp = (uint8_t*)calloc((size_t)limit + 1, 1);
    if (!is_comp) { fprintf(stderr, "alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    uint32_t r = (uint32_t)floor(sqrt((double)limit));
    for (uint32_t i = 3; i <= r; i += 2) {
        if (!is_comp[i]) {
            uint64_t step = (uint64_t)2 * i;
            uint64_t start = (uint64_t)i * i;
            for (uint64_t j = start; j <= limit; j += step) is_comp[(size_t)j] = 1;
        }
    }

    int count = 0;
    for (uint32_t i = 3; i <= limit; i += 2) if (!is_comp[i]) count++;

    uint32_t* primes = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!primes) { fprintf(stderr, "alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int idx = 0;
    for (uint32_t i = 3; i <= limit; i += 2) if (!is_comp[i]) primes[idx++] = i;

    free(is_comp);
    *count_out = count;
    return primes;
}

// Bitset helpers
static inline void bs_set(uint8_t* bits, uint64_t idx) { bits[idx >> 3] |= (uint8_t)(1u << (idx & 7u)); }
static inline int  bs_get(const uint8_t* bits, uint64_t idx) { return (bits[idx >> 3] >> (idx & 7u)) & 1u; }

static void sieve_window_and_update(
    uint64_t low, uint64_t high,
    const uint32_t* base, int base_count,
    uint64_t* local_first, uint64_t* prev_prime,
    uint64_t* local_best_gap, uint64_t* local_a, uint64_t* local_b
) {
    // Prime 2
    if (low <= 2 && 2 <= high) {
        if (*local_first == 0) *local_first = 2;
        if (*prev_prime == 0) *prev_prime = 2;
    }

    // Only odds in [max(low,3)..high]
    uint64_t seg_low = (low < 3) ? 3 : low;
    if ((seg_low & 1ull) == 0) seg_low++;
    uint64_t seg_high = high;
    if ((seg_high & 1ull) == 0) seg_high--;
    if (seg_low > seg_high) return;

    uint64_t n_odds = ((seg_high - seg_low) >> 1) + 1;
    uint64_t n_bytes = (n_odds + 7) >> 3;
    uint8_t* bits = (uint8_t*)calloc((size_t)n_bytes, 1);
    if (!bits) { fprintf(stderr, "alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // Mark composites
    for (int i = 0; i < base_count; i++) {
        uint64_t p = (uint64_t)base[i];
        uint64_t pp = p * p;
        if (pp > seg_high) break;

        uint64_t start = (seg_low + p - 1) / p * p;
        if (start < pp) start = pp;
        if ((start & 1ull) == 0) start += p; // ensure odd
        uint64_t step = 2 * p;

        for (uint64_t x = start; x <= seg_high; x += step) {
            uint64_t idx = (x - seg_low) >> 1;
            bs_set(bits, idx);
        }
    }

    // Scan primes, update gaps
    for (uint64_t idx = 0; idx < n_odds; idx++) {
        if (!bs_get(bits, idx)) {
            uint64_t prime = seg_low + (idx << 1);

            if (*local_first == 0) *local_first = prime;

            if (*prev_prime != 0) {
                uint64_t gap = prime - *prev_prime;
                if (gap > *local_best_gap) {
                    *local_best_gap = gap;
                    *local_a = *prev_prime;
                    *local_b = prime;
                }
            }
            *prev_prime = prime;
        }
    }

    free(bits);
}

static uint64_t parse_u64(const char* s, uint64_t def) {
    if (!s) return def;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return def;
    return (uint64_t)v;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t N = (argc >= 2) ? parse_u64(argv[1], 1000000000ull) : 1000000000ull;
    uint64_t W = (argc >= 3) ? parse_u64(argv[2], 100000000ull) : 100000000ull;
    if (W < 1000) W = 1000;

    if (N < 2) {
        if (rank == 0) printf("N must be >= 2\n");
        MPI_Finalize();
        return 0;
    }

    // Base primes up to sqrt(N)
    uint32_t base_limit = (uint32_t)floor(sqrt((double)N));
    uint32_t* base = NULL;
    int base_count = 0;

    if (rank == 0) base = simple_sieve_odd(base_limit, &base_count);

    MPI_Bcast(&base_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        base = (uint32_t*)malloc((size_t)base_count * sizeof(uint32_t));
        if (!base) { fprintf(stderr, "alloc fail\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(base, base_count, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Split [2..N] into contiguous chunks
    uint64_t total = (N - 2) + 1;
    uint64_t chunk = total / (uint64_t)size;
    uint64_t rem   = total % (uint64_t)size;

    uint64_t start = 2 + (uint64_t)rank * chunk + (uint64_t)((rank < (int)rem) ? rank : rem);
    uint64_t end   = start + chunk - 1;
    if ((uint64_t)rank < rem) end++;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    uint64_t local_first = 0;
    uint64_t prev_prime = 0;
    uint64_t local_best_gap = 0, local_a = 0, local_b = 0;

    for (uint64_t low = start; low <= end; ) {
        uint64_t high = low + W - 1;
        if (high > end) high = end;

        sieve_window_and_update(low, high, base, base_count,
                                &local_first, &prev_prime,
                                &local_best_gap, &local_a, &local_b);

        if (high == end) break;
        low = high + 1;
    }

    uint64_t local_last = prev_prime;

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results
    ull send_first = (ull)local_first;
    ull send_last  = (ull)local_last;
    ull send_gap   = (ull)local_best_gap;
    ull send_p1    = (ull)local_a;
    ull send_p2    = (ull)local_b;

    ull *all_first=NULL, *all_last=NULL, *all_gap=NULL, *all_p1=NULL, *all_p2=NULL;
    if (rank == 0) {
        all_first = (ull*)malloc((size_t)size * sizeof(ull));
        all_last  = (ull*)malloc((size_t)size * sizeof(ull));
        all_gap   = (ull*)malloc((size_t)size * sizeof(ull));
        all_p1    = (ull*)malloc((size_t)size * sizeof(ull));
        all_p2    = (ull*)malloc((size_t)size * sizeof(ull));
        if (!all_first||!all_last||!all_gap||!all_p1||!all_p2) {
            fprintf(stderr, "root alloc fail\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&send_first, 1, MPI_UNSIGNED_LONG_LONG, all_first, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&send_last,  1, MPI_UNSIGNED_LONG_LONG, all_last,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&send_gap,   1, MPI_UNSIGNED_LONG_LONG, all_gap,   1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&send_p1,    1, MPI_UNSIGNED_LONG_LONG, all_p1,    1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&send_p2,    1, MPI_UNSIGNED_LONG_LONG, all_p2,    1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ull best_gap = 0, best_a = 0, best_b = 0;

        // Best within any rank
        for (int i = 0; i < size; i++) {
            if (all_gap[i] > best_gap) {
                best_gap = all_gap[i];
                best_a = all_p1[i];
                best_b = all_p2[i];
            }
        }
        // Boundary gaps between ranks
        for (int i = 0; i < size - 1; i++) {
            if (all_last[i] && all_first[i + 1]) {
                ull g = all_first[i + 1] - all_last[i];
                if (g > best_gap) {
                    best_gap = g;
                    best_a = all_last[i];
                    best_b = all_first[i + 1];
                }
            }
        }

        printf("N=%llu ranks=%d window=%llu\n", (ull)N, size, (ull)W);
        printf("Largest prime gap: %llu (between %llu and %llu)\n", best_gap, best_a, best_b);
        printf("Wall time (max over ranks): %.6f seconds\n", max_time);

        free(all_first); free(all_last); free(all_gap); free(all_p1); free(all_p2);
    }

    free(base);
    MPI_Finalize();
    return 0;
}
