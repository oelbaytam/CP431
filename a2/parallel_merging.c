#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define WRITE_OUTPUT 0   // set to 1 to write merged output to a file

#define BLOCK_LENGTH 1000000LL



// Array getters, as long as the function's are scaling in ascending
static inline long long getA(long long i) { return 2LL * i; }
static inline long long getB(long long i) { return 3LL * i; }

static long long find_partition( long long sizeA, long long sizeB, long long output_rank) {
    long long low  = (output_rank > sizeB) ? (output_rank - sizeB) : 0; // either pick the starting value for the current partition or 0
    long long high = (output_rank < sizeA) ? output_rank : sizeA; //

    while (low < high) { // binary search to find partitions
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


// merging function
static void merge(
        long long sizeA, long long sizeB,
        long long *a_index, long long *b_index,
        long long *buf, long long count)
{
    long long a = *a_index;
    long long b = *b_index;

    for (long long t = 0; t < count; t++) {
        long long av = (a < sizeA) ? getA(a) : LLONG_MAX;
        long long bv = (b < sizeB) ? getB(b) : LLONG_MAX;

        if (av <= bv) { buf[t] = av; a++; }
        else { buf[t] = bv; b++; }
    }

    *a_index = a;
    *b_index = b;
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

    // find where this rank should start and end its partition of array's a and b
    long long a_start = find_partition(sizeA, sizeB, out_start);
    long long b_start = out_start - a_start;

    // create an array to merge each block
    long long *merge_array = malloc(BLOCK_LENGTH * sizeof(long long));

    #if WRITE_OUTPUT // create an array for the size of the output if file writing is desired.
        long long *local_output = malloc(output_count * sizeof(long long));
    #endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // define trackers to keep track of progress in between chunks
    long long a_cur = a_start;
    long long b_cur = b_start;
    long long produced = 0;

    //
    long long first_val = 0, last_val = 0;
    int have_data = 0;

    while (produced < output_count) {

        long long chunk = (output_count - produced > BLOCK_LENGTH) ? BLOCK_LENGTH : (output_count - produced);

        // merges chunk by chunk where a chunk is either the BLOCK_LENGTH or the remainder of values left.
        merge(
            sizeA, sizeB,
            &a_cur, &b_cur,
            merge_array,
            chunk
        );

        if (!have_data && chunk > 0) {
            first_val = merge_array[0];
            have_data = 1;
        }
        if (chunk > 0) last_val = merge_array[chunk-1];

        // write chunk data to writing output if desired
        #if WRITE_OUTPUT
            memcpy(local_output + produced,
                merge_array,
                chunk * sizeof(long long));
        #endif
        produced += chunk;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("sizeA=%lld sizeB=%lld total=%lld ranks=%d\n",
               sizeA,sizeB,total_output,world_size);
        printf("Merge time: %.6f s\n", t1-t0);

    }
    // write array to file if desired.
    #if WRITE_OUTPUT

        if (rank == 0) {
            FILE *f = fopen("merged_output.bin", "wb");

            // write current portion
            fwrite(local_output, sizeof(long long), output_count, f);

            // write all other portions
            for (int r = 1; r < world_size; r++) {
                long long count;
                MPI_Recv(&count, 1, MPI_LONG_LONG, r, 200,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                long long *tmp = malloc(count * sizeof(long long));

                MPI_Recv(tmp, count, MPI_LONG_LONG, r, 201,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                fwrite(tmp, sizeof(long long), count, f);
                free(tmp);
            }

            fclose(f);
            printf("Outputted to merged_output.bin\n");
        }
        else {
            MPI_Send(&output_count, 1, MPI_LONG_LONG, 0, 200, MPI_COMM_WORLD);
            MPI_Send(local_output, output_count, MPI_LONG_LONG, 0, 201, MPI_COMM_WORLD);
        }

    #endif

    free(merge_array);
    MPI_Finalize();
    return 0;
}
