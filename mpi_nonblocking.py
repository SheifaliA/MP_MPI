from mpi4py import MPI
import numpy as np
import time
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 3000  # Matrix size

# Rank 0 prepares data
if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    
    # Determine balanced rows
    rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
    displs = [sum(rows_per_proc[:i]) for i in range(size)]
else:
    A = None
    B = np.empty((N, N), dtype=np.float64)
    rows_per_proc = None
    displs = None

# Broadcast metadata
rows_per_proc = comm.bcast(rows_per_proc, root=0)
displs = comm.bcast(displs, root=0)

# Allocate local A chunk
local_rows = rows_per_proc[rank]
local_A = np.empty((local_rows, N), dtype=np.float64)

# Non-blocking scatter (Isend/Irecv)
if rank == 0:
    reqs = []
    for i in range(size):
        if i == 0:
            local_A[:, :] = A[displs[i]:displs[i] + rows_per_proc[i], :]
        else:
            req = comm.Isend([A[displs[i]:displs[i] + rows_per_proc[i], :], MPI.DOUBLE], dest=i, tag=77)
            reqs.append(req)
else:
    req = comm.Irecv([local_A, MPI.DOUBLE], source=0, tag=77)

# Non-blocking Bcast of matrix B
bcast_req = comm.Ibcast(B, root=0)

# Wait for communication
if rank == 0:
    MPI.Request.Waitall(reqs)
else:
    req.Wait()
bcast_req.Wait()

# Barrier to sync before timing compute
comm.Barrier()
t_start = time.perf_counter()

# Compute local dot product
local_C = np.dot(local_A, B)

comm.Barrier()
t_end = time.perf_counter()

# Prepare Gatherv
if rank == 0:
    C = np.empty((N, N), dtype=np.float64)
    recvcounts = [r * N for r in rows_per_proc]
    displs_flat = [d * N for d in displs]
else:
    C = None
    recvcounts = None
    displs_flat = None

# Gather results
comm.Gatherv(local_C, [C, recvcounts, displs_flat, MPI.DOUBLE], root=0)

# Output timing
if rank == 0:
    exec_time=t_end - t_start
    print(f"[MPI-NB-{size}] Time: {t_end - t_start:.4f} sec")
    with open("log_performance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mpi_nb", N, size, exec_time])