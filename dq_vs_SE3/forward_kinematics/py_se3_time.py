import numpy as np
from scipy.linalg import expm
import timeit
import tracemalloc
import platform
import os


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


# --- SE3 ---
def hat(v):
    # Unused
    v1, v2, v3 = v
    return np.array([
        [0,     -v3,  v2],
        [v3,   0,    -v1],
        [-v2,  v1,  0]
    ])

def form_SE3(R, d):
    g = np.zeros([4, 4])
    g[:3, :3] = R
    g[:3, 3] = d
    g[-1, -1] = 1
    return g

def expmap_se3(u):
    u1, u2, u3 = u
    u_norm = np.sqrt(u1**2 + u2**2 + u3**2)
    if u_norm < 1e-8:
        return np.eye(3)
    
    u_hat = np.array([
        [0,     -u3,  u2],
        [u3,   0,    -u1],
        [-u2,  u1,  0]
    ])

    u_hat_sq = u_hat @ u_hat
    A = np.sin(u_norm) / u_norm
    B = (1 - np.cos(u_norm)) / (u_norm**2)
    
    return np.eye(3) + A * u_hat + B * u_hat_sq


# --- Benchmark Parameters ---
num_transforms = 100
num_timeit_runs = 10000
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
        gIB = gIB @ form_SE3(expmap_se3(kappa[:, i]), d[:, i])
    return gIB



se3_time = timeit.timeit(run_se3, number=num_timeit_runs)
se3_avg = se3_time / num_timeit_runs
peak_mem_se3 = measure_peak_memory(run_se3)


print("="*50)
print("         Transformation Benchmark Results         ")
print("="*50)
print(f"{'Number of pose transformations:':<35} {num_transforms}")
print(f"{'Number of timeit runs:':<35} {num_timeit_runs}")
print(f"{'SE(3) Avg Time:':<35} {1000 * se3_avg:.3f} ms")
print(f"{'Peak Memory (SE3):':<40} {peak_mem_se3:,.1f} KB")
print("="*50)



