#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#ifndef BLOCK_ELEMS
#define BLOCK_ELEMS 1000000LL
#endif

// Array getters, as long as the function's are scaling in ascending
static inline long long getA(long long i) { return 2LL * i; }
static inline long long getB(long long i) { return 3LL * i; }


// find the j_r integers such that A_rk >= B_j(r)
static long long find_partition( long long sizeA, long long sizeB, long long output_rank) {
    long long low  = (output_rank > sizeB) ? (output_rank - sizeB) : 0;
    long long high = (output_rank < sizeA) ? output_rank : sizeA;

    while (low < high) {
        long long a_count = low + (high - low) / 2;
        long long b_count = output_rank - a_count;

        long long a_val = (a_count < sizeA) ? getA(a_count) : LLONG_MAX;
        long long b_left = (b_count > 0) ? getB(b_count - 1) : LLONG_MIN;

        if (a_val < b_left)
            low = a_count + 1;
        else
            high = a_count;
    }

    return low;
}


// -------- streaming merge with unequal bounds --------
static void stream_merge_to_buffer(
        long long sizeA,
        long long sizeB,
        long long *a_idx,
        long long *b_idx,
        long long *buf,
        long long count)
{
    long long a = *a_idx;
    long long b = *b_idx;

    for (long long t = 0; t < count; t++) {
        long long av = (a < sizeA) ? getA(a) : LLONG_MAX;
        long long bv = (b < sizeB) ? getB(b) : LLONG_MAX;

        if (av <= bv) { buf[t] = av; a++; }
        else          { buf[t] = bv; b++; }
    }

    *a_idx = a;
    *b_idx = b;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (rank == 0)
            printf("Usage: srun -n P ./merge sizeA sizeB\n");
        MPI_Finalize();
        return 1;
    }
    // Define the sizes of array A, B and Total
    long long sizeA = atoll(argv[1]);
    long long sizeB = atoll(argv[2]);
    long long total_output = sizeA + sizeB;

    // calculate the size each processor is responsible for including remainders.
    long long base = total_output / world_size;
    long long rem  = total_output % world_size;

    // define start, end, and size for each processor
    long long out_start = rank * base + (rank < rem ? rank : rem);
    long long out_end = out_start + base + (rank < rem ? 1 : 0);
    long long output_count = out_end - out_start;

    //
    long long a_start = find_partition(sizeA, sizeB, out_start);
    long long a_end   = find_partition(sizeA, sizeB, out_end);

    long long b_start = out_start - a_start;
    long long b_end   = out_end   - a_end;

    if ((a_end - a_start) + (b_end - b_start) != output_count) {
        if (rank == 0) fprintf(stderr, "Partition mismatch\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    long long *merge_buffer =
        malloc(BLOCK_ELEMS * sizeof(long long));

    if (!merge_buffer) {
        fprintf(stderr, "Rank %d malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    long long a_cur = a_start;
    long long b_cur = b_start;
    long long produced = 0;

    long long first_val = 0, last_val = 0;
    int have_data = 0;

    while (produced < output_count) {

        long long chunk =
            (output_count - produced > BLOCK_ELEMS)
            ? BLOCK_ELEMS
            : (output_count - produced);

        stream_merge_to_buffer(
            sizeA, sizeB,
            &a_cur, &b_cur,
            merge_buffer,
            chunk
        );

        if (!have_data && chunk > 0) {
            first_val = merge_buffer[0];
            have_data = 1;
        }
        if (chunk > 0) last_val = merge_buffer[chunk-1];

        produced += chunk;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // ---- neighbor boundary check ----
    long long prev_last = 0;

    if (rank > 0)
        MPI_Recv(&prev_last,1,MPI_LONG_LONG,rank-1,111,
                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    if (rank < world_size-1)
        MPI_Send(&last_val,1,MPI_LONG_LONG,rank+1,111,
                 MPI_COMM_WORLD);

    int ok_local = !(rank>0 && have_data && prev_last>first_val);

    int ok_global;
    MPI_Reduce(&ok_local,&ok_global,1,MPI_INT,
               MPI_MIN,0,MPI_COMM_WORLD);

    if (rank == 0) {
        printf("sizeA=%lld sizeB=%lld total=%lld ranks=%d\n",
               sizeA,sizeB,total_output,world_size);
        printf("Merge time: %.6f s\n", t1-t0);
        printf("Boundary check: %s\n",
               ok_global?"PASS":"FAIL");
    }

    free(merge_buffer);
    MPI_Finalize();
    return 0;
}
