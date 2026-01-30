#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --job-name mpi_job
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR

gcc prime_gap_mpi.c -o prime_gap_mpi

module load intel/2018.2
module load openmpi/3.1.0

mpirun ./prime_gap_mpi # change this file with whatever the correct application is.