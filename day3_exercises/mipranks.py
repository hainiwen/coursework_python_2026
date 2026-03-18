# Exercise 4a: MPI Ranks
# =======================
# Run with: mpirun -n 4 python mipranks.py
# Requires: pip install mpi4py
#
# Each MPI process prints its own rank and the total number of processes.

from mpi4py import MPI

comm = MPI.COMM_WORLD       # the global communicator (all processes)
rank = comm.Get_rank()      # this process's rank (0-indexed)
size = comm.Get_size()      # total number of processes

print(f"Hello from rank {rank} of {size}")

# Example output with mpirun -n 4:
# Hello from rank 0 of 4
# Hello from rank 2 of 4
# Hello from rank 1 of 4   <- order may vary!
# Hello from rank 3 of 4
