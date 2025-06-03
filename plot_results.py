import csv
import matplotlib.pyplot as plt
from collections import defaultdict

log_file = "log_performance.csv"
data = defaultdict(list)

serial_time = None

# Read and organize data
with open(log_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        method, N, processes, time_sec = row
        N = int(N)
        processes = int(processes)
        time_sec = float(time_sec)

        if method == "serial" and processes == 1:
            serial_time = time_sec
        else:
            data[method].append((processes, time_sec))

if serial_time is None:
    print("⚠️ Serial baseline not found in log.")
    exit()

# Plot speedup and efficiency
plt.figure(figsize=(12, 5))

# ---- Speedup Plot ----
plt.subplot(1, 2, 1)
for method, values in data.items():
    values.sort()
    procs = [p for p, _ in values]
    times = [t for _, t in values]
    speedup = [serial_time / t for t in times]
    label = method.upper().replace("_NB", " (Non-Blocking)").replace("MPI", "MPI (Blocking)")
    if "nonblocking" in method or "nb" in method:
        label = "MPI (Non-Blocking)"
    elif method == "mpi":
        label = "MPI (Blocking)"
    plt.plot(procs, speedup, marker="o", label=label)

plt.title("Speedup vs Number of Processes")
plt.xlabel("Processes")
plt.ylabel("Speedup")
plt.grid(True)
plt.legend()

# ---- Efficiency Plot ----
plt.subplot(1, 2, 2)
for method, values in data.items():
    values.sort()
    procs = [p for p, _ in values]
    times = [t for _, t in values]
    speedup = [serial_time / t for t in times]
    efficiency = [s / p for s, p in zip(speedup, procs)]
    label = method.upper().replace("_NB", " (Non-Blocking)").replace("MPI", "MPI (Blocking)")
    if "nonblocking" in method or "nb" in method:
        label = "MPI (Non-Blocking)"
    elif method == "mpi":
        label = "MPI (Blocking)"
    plt.plot(procs, efficiency, marker="s", label=label)

plt.title("Efficiency vs Number of Processes")
plt.xlabel("Processes")
plt.ylabel("Efficiency")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("performance_plot.png")
plt.show()
