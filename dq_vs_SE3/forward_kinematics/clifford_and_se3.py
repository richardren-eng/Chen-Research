import numpy as np

#  Generic Quaternion ==================================================================
def quatmult(q1, q2):
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

# SE(3) =================================================================================
def hat(v):
    '''
    Arg:
        v: (3, ) or (3, 1)
        
    Return: 
        v_hat: (3, 3)
    '''
    v1, v2, v3 = v
    return np.array([
        [0,     -v3,  v2],
        [v3,   0,    -v1],
        [-v2,  v1,  0]
    ])

def form_SE3(R, d):
    '''
    Args:
        R: (3, 3)
        d: (3, ) or (3, 1)

    Return:
        g: (4, 4)
    '''
    g = np.zeros([4, 4])
    g[:3, :3] = R
    g[:3, 3] = d.flatten()
    g[-1, -1] = 1
    return g

def expmap_so3(u):
    '''
    Arg: 
        u: (3, ) or (3, 1)
    
    Return:
        SO(3): (3, 3)
    '''
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

def Ad(g):
    '''
    Arg:
        g: (4, 4)

    Return:
        Ad_g: (6, 6)
    '''
    Ad_g = np.zeros((6, 6))
    R = g[:3, :3]
    Ad_g[:3, :3] = R
    Ad_g[:3, 3:] = hat(g[:3, -1]) @ R
    Ad_g[3:, 3:] = R

    return Ad_g

def expmap_se3(xi, theta):
    v, w = xi[:3], xi[3:]
    w_norm2 = np.dot(w, w)

    # rotation
    R = expmap_so3(w * theta)

    # translation
    if w_norm2 < 1e-12:        # prismatic
        d = v * theta
    else:
        w_hat = hat(w)
        d = ((np.eye(3) - R) @ (w_hat @ v) +
             np.outer(w, w) @ v * theta) / w_norm2

    return form_SE3(R, d)



# Clifford =======================================================================================
def dualquatmult(Q1, Q2):
    '''
    Args:
        Q1: (8, ) or (8, 1)
        Q2: (8, ) or (8, 1)

    Return:
        Q1Q2: (8, )
    '''
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
    '''
    Args:
        u: (3, ) or (3, 1)
        d: (3, ) or (3, 1)

    Return:
        Q: (8, )
    '''
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

def expmap_twistDQ(xi, tet):
    """
    Dual‑quaternion exponential map for a twist written with the ½‑inside convention.

    Parameters
    ----------
    xi    : (8,) ndarray
            [ 0 ,  v_x , v_y , v_z ,   0 ,  ω_x , ω_y , ω_z ]
            NB: v and ω already include the factor ½.
    theta : float
            Joint displacement (rad for revolute, metres for prismatic).

    Returns
    -------
    dq    : (8,) ndarray
            Unit dual quaternion: dq = qr  + ε qd
            qr = [w, x, y, z]   (real part, rotation)
            qd = [0, dx, dy, dz] (dual part, translation)
    """
    # ------------------------------------------------------------------
    v = xi[:4]                    # (4, )    
    omega = xi[5:]                # (3, )
 
    q = np.hstack([np.cos(tet / 2), omega * np.sin(tet/2)])           # (4, )
    r = v * tet               # (4, )
    t = 0.5 * quatmult(q, r)  # (4, )


    return np.hstack([q, t])

# Conversions ====================================================================================
def quat_to_rotmat(q):
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def dualquat_to_SE3(Q):
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
    t_quat = quatmult(qd, qr_conj)
    d = 2 * t_quat[1:]  # drop scalar part

    # Build SE(3) matrix
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = d

    return g



if __name__ == "__main__":
    # SE(3)
    u = np.array([1, 2 ,3])
    d = np.array([4, 5, 6])
    
    R = expmap_so3(u)
    g = form_SE3(R, d)
    Ad_g = Ad(g)

    # DQ
