import numpy as np
from scipy.linalg import expm
import timeit
import tracemalloc
import platform
import os

# -- Generic Quaternion --
def quat_multiply(q1, q2):
    # Unused in timing, only used in post-processing
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def expmap_to_quat(theta_vec):
    # Unused
    theta = np.linalg.norm(theta_vec)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta_vec / theta
    half_theta = 0.5 * theta
    return np.concatenate(([np.cos(half_theta)], axis * np.sin(half_theta)))



# -- Clifford --> SE(3)
def quat_to_rotmat(q):
    # Unused in timing, only used in post-processing
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def dual_quat_to_SE3(Q):
    # Unused in timing, only used in post-processing
    """
    Convert a dual quaternion to a 4x4 SE(3) transformation matrix.

    Args:
        Q: numpy array of shape (8,), where Q[:4] is the real quaternion (rotation)
            and Q[4:] is the dual quaternion (translation encoded).

    Returns:
        A 4x4 numpy array representing the SE(3) transformation matrix.
    """
    qr = Q[:4]
    qd = Q[4:]

    # Convert rotation quaternion to rotation matrix
    R = quat_to_rotmat(qr)

    # Compute translation t = 2 * (qd * conj(qr))_vector
    w, x, y, z = qr
    qr_conj = np.array([w, -x, -y, -z])
    t_quat = quat_multiply(qd, qr_conj)
    t = 2 * t_quat[1:]  # drop scalar part

    # Build SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

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


# --- Clifford ---
def dual_quat_multiply(Q1, Q2):
    rw1, rx1, ry1, rz1, tw1, tx1, ty1, tz1 = Q1
    rw2, rx2, ry2, rz2, tw2, tx2, ty2, tz2 = Q2
    return np.array([
        rw1*rw2 - rx1*rx2 - ry1*ry2 - rz1*rz2, 
        rw1*rx2 + rx1*rw2 + ry1*rz2 - rz1*ry2, 
        rw1*ry2 - rx1*rz2 + ry1*rw2 + rz1*rx2, 
        rw1*rz2 + rx1*ry2 - ry1*rx2 + rz1*rw2, 
        rw1*tw2 - rx1*tx2 - ry1*ty2 - rz1*tz2 + tw1*rw2 - tx1*rx2 - ty1*ry2 - tz1*rz2, 
        rw1*tx2 + rx1*tw2 + ry1*tz2 - rz1*ty2 + tw1*rx2 + tx1*rw2 + ty1*rz2 - tz1*ry2, 
        rw1*ty2 - rx1*tz2 + ry1*tw2 + rz1*tx2 + tw1*ry2 - tx1*rz2 + ty1*rw2 + tz1*rx2, 
        rw1*tz2 + rx1*ty2 - ry1*tx2 + rz1*tw2 + tw1*rz2 + tx1*ry2 - ty1*rx2 + tz1*rw2
        ])



def form_dual_quat(u, d):
    dx, dy, dz = d
    u_norm = np.linalg.norm(u)

    if u_norm < 1e-8:
        # Identity rotation + zero translation
        return np.array([1.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0])
    
    # Rotation quaternion
    half_u = 0.5 * u_norm
    cos_hu = np.cos(half_u)
    rx, ry, rz = u / u_norm * np.sin(half_u)

    return np.array([
        cos_hu, 
        rx, 
        ry, 
        rz, 
        -0.5 * (dx * rx + dy * ry + dz * rz), 
        0.5 * (cos_hu * dx + dy * rz - dz * ry), 
        0.5 * (cos_hu * dy + dz * rx - dx * rz), 
        0.5 * (cos_hu * dz + dx * ry - dy * rx)
        ])



# --- Benchmark Parameters ---
num_transforms = 100
num_timeit_runs = 10000
seed = 807
test_DQ_first = 1
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
