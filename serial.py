import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import csv
from mpi4py import MPI
import csv
import subprocess

def run_serial(N):
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    start = time.perf_counter()
    np.dot(A, B)
    end = time.perf_counter()
    return end - start
