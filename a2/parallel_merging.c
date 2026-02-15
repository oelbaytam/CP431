#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>


#if defined(_MSC_VER)
  #include <malloc.h>
  #define aligned_free _aligned_free
#else
  #define aligned_free free
#endif

// ----------- Tunables -----------
#define ALIGN_BYTES 64

#ifndef DO_GATHER_RESULT
#define DO_GATHER_RESULT 1
#endif

#ifndef TIME_MERGE_ONLY
#define TIME_MERGE_ONLY 1
#endif


static void *aligned_malloc(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, ALIGN_BYTES);
#else
    void *p = NULL;
    if (posix_memalign(&p, ALIGN_BYTES, bytes) != 0) return NULL;
    return p;
#endif
}

static inline void sequential_merge_ptr(const int *a, int n1, const int *b, int n2, int *out) {
    const int *pa = a, *pb = b;
    const int *ea = a + n1, *eb = b + n2;
    int *po = out;

    while (pa < ea && pb < eb) {
        *po++ = (*pa <= *pb) ? *pa++ : *pb++;
    }

    if (pa < ea) {
        size_t rem = (size_t)(ea - pa);
        memcpy(po, pa, rem * sizeof(int));
        po += rem;
    }
    if (pb < eb) {
        size_t rem = (size_t)(eb - pb);
        memcpy(po, pb, rem * sizeof(int));
    }
}


static int find_partition_safe(const int *arr1, int n1, const int *arr2, int n2, int target_pos) {

    int low = (target_pos - n2 > 0) ? (target_pos - n2) : 0;
    int high = (target_pos < n1) ? target_pos : n1;

    while (low < high) {
        int i = low + (high - low) / 2;
        int j = target_pos - i;

        if (i < n1 && j > 0 && arr1[i] < arr2[j - 1]) {
            low = i + 1;
        } else {
            high = i;
        }
    }
    return low;
}

int main(int argc, char *argv[]) {
    int rank, size;
    long long n1, n2;

    if (argc < 2) {
        if (rank == 0)
            printf("Usage: mpirun -np <procs> ./Parallel_merge <num_elements_per_array>\n");
        MPI_Finalize();
        return 1;
    }

    n1 = atoll(argv[1]);
    n2 = n1;   // same size for both arrays


    int *arr1 = NULL, *arr2 = NULL, *result = NULL;

    int *s_counts1 = NULL, *displs1 = NULL;
    int *s_counts2 = NULL, *displs2 = NULL;
    int *recv_counts = NULL, *displs_res = NULL;

    int local_n1 = 0, local_n2 = 0;
    int *l_arr1 = NULL, *l_arr2 = NULL, *l_res = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        arr1 = (int *)aligned_malloc((size_t)n1 * sizeof(int));
        arr2 = (int *)aligned_malloc((size_t)n2 * sizeof(int));
        if (!arr1 || !arr2) {
            fprintf(stderr, "Root: allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Running with %lld elements per array (%lld total)\n", n1, n1 + n2);



#if DO_GATHER_RESULT
        result = (int *)aligned_malloc((size_t)(n1 + n2) * sizeof(int));
        if (!result) {
            fprintf(stderr, "Root: result allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#endif

        // Generate sorted inputs
        for (long long i = 0; i < n1; i++) arr1[i] = (int)(i * 2);
        for (long long i = 0; i < n2; i++) arr2[i] = (int)(i * 2 + 1);

        s_counts1 = (int *)malloc((size_t)size * sizeof(int));
        displs1   = (int *)malloc((size_t)size * sizeof(int));
        s_counts2 = (int *)malloc((size_t)size * sizeof(int));
        displs2   = (int *)malloc((size_t)size * sizeof(int));
        recv_counts = (int *)malloc((size_t)size * sizeof(int));
        displs_res  = (int *)malloc((size_t)size * sizeof(int));

        if (!s_counts1 || !displs1 || !s_counts2 || !displs2 || !recv_counts || !displs_res) {
            fprintf(stderr, "Root: partition arrays allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        long long total_ll = n1 + n2;

        if (total_ll / size > INT_MAX) {
            if (rank == 0)
                printf("Error: Each process would exceed MPI 32-bit count limit.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int total = (int) total_ll;

        const int chunk = total / size;
        const int rem   = total % size;

        for (int r = 0; r < size; r++) {
            const int s_idx = r * chunk + (r < rem ? r : rem);
            const int e_idx = s_idx + chunk + (r < rem ? 1 : 0);

            const int p1_s = find_partition_safe(arr1, (int)n1, arr2, (int)n2, s_idx);
            const int p1_e = find_partition_safe(arr1, (int)n1, arr2, (int)n2, e_idx);

            const int p2_s = s_idx - p1_s;
            const int p2_e = e_idx - p1_e;

            s_counts1[r] = p1_e - p1_s;
            displs1[r]   = p1_s;

            s_counts2[r] = p2_e - p2_s;
            displs2[r]   = p2_s;

            recv_counts[r] = e_idx - s_idx;
            displs_res[r]  = s_idx;
        }
    }

    MPI_Scatter(s_counts1, 1, MPI_INT, &local_n1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(s_counts2, 1, MPI_INT, &local_n2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    l_arr1 = (int *)aligned_malloc((size_t)local_n1 * sizeof(int));
    l_arr2 = (int *)aligned_malloc((size_t)local_n2 * sizeof(int));
    l_res  = (int *)aligned_malloc((size_t)(local_n1 + local_n2) * sizeof(int));

    if ((!l_arr1 && local_n1 > 0) || (!l_arr2 && local_n2 > 0) || (!l_res && (local_n1 + local_n2) > 0)) {
        fprintf(stderr, "Rank %d: local allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(arr1, s_counts1, displs1, MPI_INT,
                 l_arr1, local_n1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(arr2, s_counts2, displs2, MPI_INT,
                 l_arr2, local_n2, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    sequential_merge_ptr(l_arr1, local_n1, l_arr2, local_n2, l_res);

#if TIME_MERGE_ONLY
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
#else
    // You can also time end-to-end including gather below if desired.
    double t1 = MPI_Wtime();
#endif

    // --- Optional gather result (expensive) ---
#if DO_GATHER_RESULT
    MPI_Gatherv(l_res, local_n1 + local_n2, MPI_INT,
                result, recv_counts, displs_res, MPI_INT,
                0, MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank == 0) {
        if (TIME_MERGE_ONLY) {
            printf("Parallel local-merge time (barrier-bounded): %f s\n", t1 - t0);
        } else {
            printf("Time after scatter (includes merge pre-gather): %f s\n", t1 - t0);
        }

#if DO_GATHER_RESULT
        printf("Total time incl gather (barrier-bounded): %f s\n", t2 - t0);
        printf("Processing %lld total elements...\n", n1 + n2);

        int sorted = 1;
        for (long long i = 1; i < (n1 + n2); i++) {
            if (result[i] < result[i - 1]) { sorted = 0; break; }
        }
        printf("Sorted: %s\n", sorted ? "YES" : "NO");
#else
        printf("Result gather skipped (DO_GATHER_RESULT=0).\n");
#endif
    }

    aligned_free(l_arr1);
    aligned_free(l_arr2);
    aligned_free(l_res);

    if (rank == 0) {
        aligned_free(arr1);
        aligned_free(arr2);
#if DO_GATHER_RESULT
        aligned_free(result);
#endif
        free(s_counts1); free(displs1);
        free(s_counts2); free(displs2);
        free(recv_counts); free(displs_res);
    }

    MPI_Finalize();
    return 0;
}


