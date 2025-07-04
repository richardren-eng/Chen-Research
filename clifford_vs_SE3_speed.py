import numpy as np
from scipy.linalg import expm
import timeit

# -- Generic Quaternion --
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def expmap_to_quat(theta_vec):
    theta = np.linalg.norm(theta_vec)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta_vec / theta
    half_theta = 0.5 * theta
    return np.concatenate(([np.cos(half_theta)], axis * np.sin(half_theta)))



# -- Clifford --> SE(3)
def quat_to_rotmat(q):
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def dual_quat_to_SE3(Q):
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



# --- SE3 ---
def hat(v):
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
    return expm(np.array([
        [0,     -u3,  u2],
        [u3,   0,    -u1],
        [-u2,  u1,  0]
    ]))


# --- Clifford ---
def dual_quat_multiply(Q1, Q2):
    # Real parts
    rw1, rx1, ry1, rz1 = Q1[:4]
    rw2, rx2, ry2, rz2 = Q2[:4]

    # Dual parts
    tw1, tx1, ty1, tz1 = Q1[4:]
    tw2, tx2, ty2, tz2 = Q2[4:]

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

def form_dual_quat(theta_vec, d):
    theta = np.linalg.norm(theta_vec)
    
    if theta < 1e-8:
        # Identity rotation + zero translation
        return np.array([1.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0])
    
    # Rotation quaternion
    rot_axis = theta_vec / theta
    half_theta = 0.5 * theta
    sin_ht = np.sin(half_theta)
    cos_ht = np.cos(half_theta)
    rx, ry, rz = rot_axis * sin_ht

    # Translation vector
    dx, dy, dz = d

    # Compute q_d = 0.5 * [0, d] * q_r (inlined quaternion multiplication)
    qw = -0.5 * (dx * rx + dy * ry + dz * rz)
    qx =  0.5 * (cos_ht * dx + dy * rz - dz * ry)
    qy =  0.5 * (cos_ht * dy + dz * rx - dx * rz)
    qz =  0.5 * (cos_ht * dz + dx * ry - dy * rx)

    return np.array([cos_ht, rx, ry, rz, qw, qx, qy, qz])



# --- Data generation ---
N = 100
seed = 56

# The curvature and displacement is obtained at the start of each step
# --- Timing SE(3) multiplication ---
def run_se3():
    np.random.seed(seed)
    gIB = np.eye(4)

    # Simulate forward marching
    for i in range(N):
        kappa = np.random.rand(3)
        d = np.random.rand(3)
        gIB = gIB @ form_SE3(expmap_se3(kappa), d)
    return gIB

# --- Timing dual quaternion multiplication ---
def run_dualquat():
    np.random.seed(seed)
    QIB = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Simulate forward marching
    for i in range(N):
        kappa = np.random.rand(3)
        d = np.random.rand(3)
        QIB = dual_quat_multiply(QIB, form_dual_quat(kappa, d))
    return QIB

# --- Timeit ---
num_runs = 100
se3_time = timeit.timeit(run_se3, number=num_runs)
Q_time = timeit.timeit(run_dualquat, number=num_runs)

se3_avg = se3_time / num_runs
Q_avg = Q_time / num_runs

gIB = run_se3()
QIB = run_dualquat()

zero_tol = 1e-8
assert np.linalg.norm(gIB - dual_quat_to_SE3(QIB), ord='fro') < zero_tol
print(f"Number of transformations: {N}")
print(f"Number of timeit runs: {num_runs}")
print(f"SE3 Avg (ms): {1000 * se3_avg} ")
print(f"Dual Quaternion Avg (ms): {1000 * Q_avg}")

