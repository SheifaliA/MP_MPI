from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 3000  # Matrix size

# Rank 0 creates matrices
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    # Compute even distribution of rows
    rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
    displs = [sum(rows_per_proc[:i]) for i in range(size)]
else:
    A = None
    B = np.empty((N, N), dtype=np.float64)
    rows_per_proc = None
    displs = None

# Broadcast B to all
comm.Bcast(B, root=0)

# Distribute row counts to all ranks
rows_per_proc = comm.bcast(rows_per_proc, root=0)
displs = comm.bcast(displs, root=0)

# Each process allocates its part of A
local_rows = rows_per_proc[rank]
local_A = np.empty((local_rows, N), dtype=np.float64)

# Prepare Scatterv args
if rank == 0:
    sendbuf = [A, tuple(np.array(rows_per_proc) * N), tuple(np.array(displs) * N), MPI.DOUBLE]
else:
    sendbuf = None

# Scatter rows of A
comm.Scatterv(sendbuf, local_A, root=0)

# Time compute
comm.Barrier()
t_start = time.perf_counter()

local_C = np.dot(local_A, B)

comm.Barrier()
t_end = time.perf_counter()

# Gather results
if rank == 0:
    C = np.empty((N, N), dtype=np.float64)
    recvbuf = [C, tuple(np.array(rows_per_proc) * N), tuple(np.array(displs) * N), MPI.DOUBLE]
else:
    recvbuf = None

comm.Gatherv(local_C, recvbuf, root=0)
import csv
# Log timing
if rank == 0:
    total_time=t_end - t_start
    print(f"[MPI-{size}] Time: {t_end - t_start:.4f} sec")
    with open("log_performance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mpi", N, size, total_time])
