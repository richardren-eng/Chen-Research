from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Math functions
def hat(v: np.ndarray) -> np.ndarray:
    """
    Casts 3 vector into a so(3) matrix
    """
    return np.array([[0, -v[2],  v[1]],
                     [v[2],   0, -v[0]],
                     [-v[1],  v[0],  0]])


def exp_se3(xi: np.ndarray, ds: float) -> np.ndarray:
    """Transforms a 6 vector twist into a SE(3) matrix"""
    w, v = xi[:3], xi[3:]
    th = np.linalg.norm(w)  

    # Pure translation
    if th < 1e-9:                       
        R = np.eye(3)
        p = v * ds

    # Screw with rotation and translation
    else:
        R = expm(hat(w) * ds)          
        V = (ds * I
             + (1 - np.cos(th*ds)) / th**2 * hat(w)
             + (th*ds - np.sin(th*ds)) / th**3 * hat(w) @ hat(w))
        p = V @ v

    gmat = np.eye(4)
    gmat[:3,:3] = R
    gmat[:3, 3] = p
    return gmat


def difference(edge: np.ndarray) -> np.ndarray:
    """
    Central finite difference operator
    """
    N = len(edge)
    out = np.zeros((N+1, 3))
    out[0]      = -edge[0]
    out[1:-1]   =  edge[1:] - edge[:-1] 
    out[-1]     = -edge[-1]
    return out


def average(edge: np.ndarray) -> np.ndarray:
    """
    Trapezoidal average operator
    """
    N = len(edge)
    out = np.zeros((N+1, 3))
    out[0]    = 0.5 * edge[0]
    out[1:-1] = 0.5 * (edge[:-1] + edge[1:])
    out[-1]   = 0.5 * edge[-1]
    return out



# Discretized Residual (static equilibrium)
def build_residual(x, *, N, ds, S, B, sigma0, kappa0,
                   r0, directors0,
                   F_tip, C_tip):
    
    """
    Return the residual vector f(x) with length 6(N+1) and 3 equations.

    args: 
    x:           unknown vector 
    N:           number of segments i.e. edges to use for the discretization. There will be N + 1 vertices i.e. nodes.
    ds:          length of each discrete segment i.e. edge
    S:           stretch and shear matrix resolved in the body frame
    B:           bending and torsion matrix resolved in the body frame
    sigma0:      zero-stress strain configuration
    kappa0:      zero-stress curvature configuration
    r0:          location of the base of the rod relative to the inertial frame's origin resolved in the inertial frame
    directors0:  rotation matrix of the body frame relative to the inertial frame at the rod's base
    F_tip:       force on the free tip of the robot resolved in the ??? frame
    C_tip:       couple on the free tip of the rod resolved in the ??? frame
    """

    # Extract the values from the unknown vector x
    # n0     = entries 0 - 2
    # c0     = entries 3 - 5
    # sigma0 = entries 6 - 8
    # sigma1 = entries 9 - 11
    # sigma2 = entries 12 - 14
    # ...
    # kappa0 = entries 6+3N‒1 ...  

    idx = 0
    n0 = x[idx:idx+3];  idx += 3
    c0 = x[idx:idx+3];  idx += 3
    sigma = x[idx:idx+3*N].reshape(N, 3); idx += 3*N
    kappa = x[idx:idx+3*N].reshape(N, 3)

    # Constitutive laws to get the internal forces and couples for each discrete segment
    n_edge = (S @ (sigma - sigma0).T).T         # N×3
    c_edge = (B @ (kappa - kappa0).T).T         # N×3

    # Initial pose (SO(3)) at the root of the rod
    g = np.eye(4)

    # g_{I0} which takes the inertial frame to the 0th body frame
    g[:3, 3] = r0
    g[:3, :3] = directors0

    # List of the vertices i.e. nodes and edges i.e. segments
    verts = [g[:3, 3]]
    l = []

    # Propogate discrete kinematics throughout the length of the rod starting from the root
    for i in range(N):
        g = g @ exp_se3(np.hstack([kappa[i], sigma[i]]), ds) # this is g_{I -> i + 1}   kappa[i] = [kappa_{i -> i + 1}]_i
        l.append(g[:3, 3] - verts[-1]) # this is [l_i]_I
        verts.append(g[:3, 3]) # this is [r_I -> i]_I

    l = np.array(l)               # list of l_i where i = 0, 1, ..., 3N 
    
    # voronoi domains
    D = np.linalg.norm(l.reshape(-1, 3), axis = 1)

    # Vertex‑wise force balance   Delta^h n = 0  (interior)  + boundary terms  
    force_vtx = difference(n_edge)

    # Vertex‑wise moment balance   Delta^h m + (average l) x n = 0
    lever = average(l)                   # half‑edges at vertices
    moment_vtx = difference(c_edge) + np.cross(lever, force_vtx)

    # Build the residual
    res = []

    # interior vertices (1 … N‑1) 
    # The force from i must cancel the negative of the force from i + 1
    res += list(force_vtx[1:-1].ravel())
    res += list(moment_vtx[1:-1].ravel())

    # Root is clamped: position and directors at the root are fixed 
    res += list(verts[0] - r0)                  # 3 eqns to enforce the root position is the specified root position
    R_root = directors0
    res += list((R_root.T @ R_root - np.eye(3)).flatten()[:3])  # 3 eqns to enfore that the root orientation matrix is SO(3)

    # Tip is free: internal wrench must balance the applied wrench
    res += list(force_vtx[-1] + F_tip)          # 3 eqns
    res += list(moment_vtx[-1] + C_tip)         # 3 eqns
    return np.array(res)



 
if __name__ == "__main__":
    # mMterial properties
    L  = 0.3              # m
    E, G = 1.0e9, 0.5e9   # Pa
    A  = 1.0e-4           # m^2
    I  = 1.0e-8           # m^4
    J  = 2*I
    S = np.diag([G*A, G*A, E*A])
    B = np.diag([E*I, E*I, G*J])

    # Discretization
    N  = 50
    ds = L / N

    # Zero-stress strain and curvature
    sigma0 = np.array([0, 0, 1])   # unstretched, no shear
    kappa0 = np.zeros(3)           # straight

    # Boundary conditions
    r_root = np.zeros(3)
    directors_root = np.eye(3)     # clamp aligned with inertial frame

    # Applied force at the free end of the rod
    F_tip = np.array([0, 0, 0])
    C_tip = np.array([0, 0, 0])  

    # unknown vector (6 + 6N)
    x0 = np.zeros(6 + 6*N)
    x0[2] = -0.1            # small axial tension guess so Jacobian non‑singular

    # Solve nonlinear system
    sol = root(
            lambda x: build_residual(x,
                                    N = N, ds = ds, S = S, B = B,
                                    sigma0 = sigma0, kappa0 = kappa0,
                                    r0 = r_root, directors0 = directors_root,
                                    F_tip = F_tip, C_tip = C_tip),
            x0,
            method="lm",
            tol=1e-4,
            options={"maxiter": 10000}
    )


    if not sol.success:
        raise RuntimeError(sol.message)

    # reconstruct centrline for plotting
    idx = 0
    n0 = sol.x[idx:idx+3];  idx += 3
    c0 = sol.x[idx:idx+3];  idx += 3
    sigma = sol.x[idx:idx+3*N].reshape(N, 3); idx += 3*N
    kappa = sol.x[idx:idx+3*N].reshape(N, 3)

    g = np.eye(4)
    g[:3, 3] = r_root
    g[:3, :3] = directors_root

    pts = [g[:3, 3]]
    for i in range(N):
        g = g @ exp_se3(np.hstack([kappa[i], sigma[i]]), ds)
        pts.append(g[:3, 3])

    pts = np.asarray(pts)

    # Plot solved shape
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", lw=2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Static Cosserat rod – N={N} segments")
    ax.set_aspect('equal')
    plt.tight_layout(); plt.show()
    
