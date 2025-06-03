import os
import csv
import time
import subprocess
import numpy as np
from serial import run_serial
N = 3000
log_file = "log_performance.csv"

# Remove old log file if exists
if os.path.exists(log_file):
    os.remove(log_file)

print("Running serial test...")
serial_time = run_serial(N)
print(f"[SERIAL] Time: {serial_time:.4f} sec")

with open(log_file, "a", newline="") as f:
    csv.writer(f).writerow(["serial", N, 1, serial_time])

# MPI test configurations
mpi_tests = {
    "blocking": "mpi_blocking.py",
    "nonblocking": "mpi_nonblocking.py"
}

process_counts = [4, 8, 10,12]

for impl_name, script_name in mpi_tests.items():
    for p in process_counts:
        print(f"Running MPI ({impl_name}) with {p} process(es)...")

        result = subprocess.run(
             ["mpiexec","-n", str(p), "python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )

        print(result.stdout)
        if result.stderr:
            print(f"MPI Errors ({impl_name}, {p} procs):\n", result.stderr)

        # Parse execution time from output, expects line like:
        # [MPI-<size>] Time: X.XXXX sec
        exec_time = None
        for line in result.stdout.splitlines():
            if line.startswith(f"[MPI-{p}]") and "Time:" in line:
                try:
                    exec_time = float(line.strip().split()[-2])
                except Exception as e:
                    print(f"Failed to parse time ({impl_name}, {p} procs): {e}")
                break

        if exec_time is not None:
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([impl_name, N, p, exec_time])
        else:
            print(f"No valid execution time found for {impl_name} with {p} procs.")

print("All tests completed. Performance logged in", log_file)
