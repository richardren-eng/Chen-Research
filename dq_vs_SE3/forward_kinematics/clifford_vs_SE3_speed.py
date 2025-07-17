import numpy as np
from scipy.linalg import expm
import timeit
import tracemalloc
import platform
import os
from clifford_and_se3 import *

# --- Memory Usage Tracking ---
def measure_peak_memory(func):
    '''
    Return:
        KB of RAM consumed
    '''
    tracemalloc.start()
    func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024  # KB


# --- System Information ---
def get_system_info():
    return {
        "Processor": platform.processor(),
        "Machine": platform.machine(),
        "Architecture": platform.architecture()[0],
        "OS": platform.system() + " " + platform.release()
    }


# --- Benchmark Parameters ---
num_transforms = 100
num_timeit_runs = 1000
seed = 807
test_DQ_first = 0
zero_tol = 1e-8

# The curvature and displacement are obtained at the start of each step
kappa = np.random.rand(3, num_transforms)
d = np.random.rand(3, num_transforms)

# --- Timing SE(3) multiplication ---
def run_se3():
    gIB = np.eye(4)
    # Simulate forward marching
    for i in range(num_transforms):
        gIB = gIB @ form_SE3(expmap_so3(kappa[:, i]), d[:, i])
    return gIB

# --- Timing dual quaternion multiplication ---
def run_dualquat():
    np.random.seed(seed)
    QIB = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Simulate forward marching
    for i in range(num_transforms):
        QIB = dual_quat_multiply(QIB, form_dual_quat(kappa[:, i], d[:, i]))
    return QIB

# --- Timeit and Memory Usage Tests---
if test_DQ_first:
    Q_time = timeit.timeit(run_dualquat, number=num_timeit_runs)
    Q_avg = Q_time / num_timeit_runs
    peak_mem_dq = measure_peak_memory(run_dualquat)
    se3_time = timeit.timeit(run_se3, number=num_timeit_runs)
    se3_avg = se3_time / num_timeit_runs
    peak_mem_se3 = measure_peak_memory(run_se3)
else:
    se3_time = timeit.timeit(run_se3, number=num_timeit_runs)
    se3_avg = se3_time / num_timeit_runs
    peak_mem_se3 = measure_peak_memory(run_se3)
    Q_time = timeit.timeit(run_dualquat, number=num_timeit_runs)
    Q_avg = Q_time / num_timeit_runs
    peak_mem_dq = measure_peak_memory(run_dualquat)

# Ensure both methods produced the same final pose
QIB = run_dualquat()
gIB = run_se3()
assert np.linalg.norm(gIB - dual_quat_to_SE3(QIB), ord='fro') < zero_tol

print("="*50)
print("         Transformation Benchmark Results         ")
print("="*50)
print(f"{'Order of Testing:':<35} {'DQ, SE(3)' if test_DQ_first else 'SE(3), DQ'}")
print(f"{'Number of pose transformations:':<35} {num_transforms}")
print(f"{'Number of timeit runs:':<35} {num_timeit_runs}")
print(f"{'SE(3) Avg Time:':<35} {1000 * se3_avg:.3f} ms")
print(f"{'Dual Quaternion Avg Time:':<35} {1000 * Q_avg:.3f} ms")
print(f"{'DQ is how many × faster than SE3:':<35} {se3_avg / Q_avg:.2f}× faster")
print(f"{'Peak Memory (SE3):':<40} {peak_mem_se3:,.1f} KB")
print(f"{'Peak Memory (Dual Quaternion):':<40} {peak_mem_dq:,.1f} KB")
print(f"{'Memory saving (DQ vs SE3):':<40} {peak_mem_se3 / peak_mem_dq:,.2f}× less memory")
print("="*50)
sysinfo = get_system_info()
print(f"{'Processor:':<35} {sysinfo['Processor']}")
print(f"{'Machine:':<35} {sysinfo['Machine']}")
print(f"{'Architecture:':<35} {sysinfo['Architecture']}")
print(f"{'OS:':<35} {sysinfo['OS']}")


