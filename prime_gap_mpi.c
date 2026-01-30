#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************
CP431 Assignment 1
---------------
Biggest potential prime is n, the biggest possible factor of n is sqrt(n)
Calculate all the primes up to sqrt(n) and call them "primes"

@author: Brandon Dang, Connor Doidge, Jackson Dow, Omar El-Baytam, Kerem Erkoc
 ***************************************************************/

#define SEGMENT_SIZE 1000000 // Defined 1 million but the theory is that if its small it fits within the CPU Cache

/***************************************************************
@method:
simple_sieve
---------------
computes all prime numbers up to sqrt(n) using the Sieve of
Eratosthenes

@parameter:
sqrt_n - square root of the upper bound n
prime_count - output pointer storing number of primes found

@return:
pointer to array containing all primes <= sqrt(n)
***************************************************************/
int* simple_sieve(long sqrt_n, int *prime_count) { // Finds prime numbers up to sqrt(n) to use in the segmented sieve.
    char *is_prime = (char *)malloc(sqrt_n + 1); // Create a character array of primes up to sqrt(n) + 1
    memset(is_prime, 1, sqrt_n + 1); // Assume all characters are primes.
    is_prime[0] = is_prime[1] = 0; // 0 and 1 wont be used to calculate other primes.

    for (long i = 2; i * i <= sqrt_n; i++) { // A simple seive from 2 to sqrt(n)

        if (is_prime[i]) { // if a prime is found

            for (long j = i * i; j <= sqrt_n; j += i) is_prime[j] = 0; // mark all multiples to sqrt(n) as non primes.

        }

    }

    // calculate amount of primes to use in prime array definition
    int count = 0;
    for (long i = 2; i <= sqrt_n; i++) if (is_prime[i]) count++;

    // define array of primes to use in each segment
    int *primes = (int *)malloc(count * sizeof(int));
    int index = 0;
    for (long i = 2; i <= sqrt_n; i++) { // loop through all values from 2 to sqrt(n)

        if (is_prime[i]) primes[index++] = (int)i; // if its a prime add it to primes[index]

    }

    free(is_prime);
    *prime_count = count;
    return primes;
}

/***************************************************************
@method:
simple_sieve
---------------
Sieve a specific segment and update gap statistics

@parameter:
seg_low - lower bound 
seg_high - upper bound
primes - array of primes <= sqrt(n)
prime_count - number of primes
local_max - largest prime gap
prep_p - previous prime encountered
first_p - first prime found
last_p - last prime found

@return:
None
 ***************************************************************/
void sieve_segment(long seg_low, long seg_high, int *primes, int prime_count,
                   long *local_max, long *prev_p, long *first_p, long *last_p) {

    int range = (int)(seg_high - seg_low + 1);
    char *segment = (char *)malloc(range);
    memset(segment, 0, range); // 0 is prime

    for (int i = 0; i < prime_count; i++) {

        long p = primes[i];
        // Calculate the first multiple of p within this segment
        long start = (seg_low + p - 1) / p * p;

        if (start < p * p) start = p * p; // Don't cross out the prime prime itself
        for (long j = start; j <= seg_high; j += p) {

            segment[j - seg_low] = 1; // Mark as composite
        }

    }

    // Process the results of this segment
    for (int i = 0; i < range; i++) {

        if (!segment[i]) {

            long curr = seg_low + i;
            if (*first_p == -1) *first_p = curr;
            if (*prev_p != -1) {
                long gap = curr - *prev_p;
                if (gap > *local_max) *local_max = gap;
            }
            *prev_p = curr;
            *last_p = curr;

        }

    }

    free(segment);

}

// --- MAIN MPI COORDINATOR ---
int main(int argc, char *argv[]) {
    int id, p;
    long n;
    double start_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (id == 0) printf("Usage: %s <n>\n", argv[0]);
        MPI_Finalize(); exit(1);
    }
    n = atoll(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 1. Generate/Broadcast primes
    int prime_count;
    int *primes = NULL;
    if (id == 0) primes = simple_sieve((long)sqrt((double)n), &prime_count);

    MPI_Bcast(&prime_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (id != 0) primes = (int *)malloc(prime_count * sizeof(int));
    MPI_Bcast(primes, prime_count, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Define Local Range
    long low_bound = 2 + (long)id * (n - 1) / p;
    long high_bound = 1 + (long)(id + 1) * (n - 1) / p;

    long local_max_gap = 0, prev_p = -1, first_p = -1, last_p = -1;

    // 3. Segmented Loop
    for (long curr_low = low_bound; curr_low <= high_bound; curr_low += SEGMENT_SIZE) {

        long curr_high = curr_low + SEGMENT_SIZE - 1;

        if (curr_high > high_bound) curr_high = high_bound;

        sieve_segment(curr_low, curr_high, primes, prime_count,
                      &local_max_gap, &prev_p, &first_p, &last_p);

    }

    // 4. Consolidate Results
    long global_max_gap = 0;
    long *all_firsts = (id == 0) ? (long *)malloc(p * sizeof(long)) : NULL;
    long *all_lasts  = (id == 0) ? (long *)malloc(p * sizeof(long)) : NULL;

    MPI_Gather(&first_p, 1, MPI_LONG_LONG, all_firsts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&last_p, 1, MPI_LONG_LONG, all_lasts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_gap, &global_max_gap, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (id == 0) {

        // Check gaps between process boundaries
        for (int i = 0; i < p - 1; i++) {

            if (all_lasts[i] != -1 && all_firsts[i+1] != -1) {

                long bridge = all_firsts[i+1] - all_lasts[i];
                if (bridge > global_max_gap) global_max_gap = bridge;

            }

        }

        printf("Result for n = %ld: Largest Gap = %ld\n", n, global_max_gap);
        printf("Time: %f seconds using %d processes\n", MPI_Wtime() - start_time, p);
        free(all_firsts); free(all_lasts);

    }

    free(primes);
    MPI_Finalize();
    return 0;

}