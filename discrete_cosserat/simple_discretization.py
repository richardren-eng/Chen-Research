import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import root, least_squares
from matplotlib import pyplot as plt

# At the top of your script (optional)
debug_vars = {}

# Linear Algebra Functions
def _hat(v):
    '''
    Converts a 3-vector to its corresponding so(3) matrix.
    
    Args:
        v: size 3 np.array 

    Returns:
        v_hat: (3,3) np.array 
    '''
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

def _vee(v_hat):
    '''
    Converts a so(3) matrix to its corresponding 3-vector.

    Args:
        v_hat: so(3) matrix (3, 3) np.array

    Returns:
        v: (3,) np.array
    '''
    return np.array([
        v_hat[2,1],
        v_hat[0,2],
        v_hat[1,0]
    ])

def _unpack(q):
    '''
    Arg:
        q = [n1.T, m1.T, r2.T, r3.T, ..., r_{N+1}.T, Theta2.T, Theta3.T, ..., Theta_{N+1}.T] (6(N+1), )
    
        Return
    '''
    N = int(int(q.size)/6 - 1)
    n1 = q[:3]                                     # (3, )
    m1 = q[3:6]                                    # (3, )
    rD = q[6: 6+3*N].reshape(3, -1, order='F')     # (3, N)
    ThetaD = q[6+3*N:].reshape(3, -1, order='F')   # (3, N)

    return n1, m1, rD, ThetaD

def _residual(q, params):
    N = params["N"]                                # Scalar
    ds = params["ds"]                              
    A = params["Area"]
    E = params["Young"]
    I = params["I"]
    f_ext = params["f_ext"]                        # (3, N)
    c_ext = params["c_ext"]                        # (3, N)
    r1 = params["r_base"].reshape(3, -1)           # (3, 1)
    Theta1 = params["Theta_base"].reshape(3, -1)   # (3, 1)
    SB_inv = params["SB_inv"]                      # (3, 3)
    BB_inv = params["BB_inv"]                      # (3, 3)
    F_tip = params["F_tip"]                        # (3, )
    M_tip = params["M_tip"]                        # (3, )


    # Extract unknowns
    n1, m1, rD, ThetaD = _unpack(q)

    # Form full collection of vertices and directors where the base quantites are known
    r = np.hstack([r1, rD])                  # (3, N+1)
    Theta = np.hstack([Theta1, ThetaD])      # (3, N+1)

    # Discrete rotation matrices
    RIB = np.zeros([3, 3*(N+1)])             # (3, 3(N+1))
    for i in range(N+1):
        RIB[:, 3*i : 3*(i+1)] = expm(_hat(Theta[:, i]))

    # Discrete internal forces
    n = np.zeros([3, N+1])                   # (3, N+1)
    n[:, 0] = n1
    for i in range(N):
        n[:, i+1] = n[:, i] - ds * f_ext[:, i]

    # Discrete internal moments
    m = np.zeros([3, N+1])
    m[:, 0] = m1
    for i in range(N):
        m[:, i+1] = m[:, i] - ds * (np.linalg.cross((r[:, i+1] - r[:, i]) / ds, n[:, i]) + c_ext[:, i])
    
    # Discrete positions and directors
    res_pos = np.zeros([3, N+1])
    res_directors = np.zeros([3, N+1])
    for i in range(N):
        Ri = RIB[:, 3*i : 3*(i+1)]
        Rip1 = RIB[:, 3*(i+1) : 3*(i+2)]
        res_pos[:,i] = -r[:, i+1] + r[:, i] + ds * Ri @ (np.array([0, 0, 1]) + SB_inv @ Ri.T @ n[:, i])
        res_directors[:, i] = _vee(logm(Ri.T @ Rip1)) - ds * BB_inv @ Ri @ m[:, i]

    # Tip boundary condition
    res_tip_force = n[:, -1] + F_tip
    res_tip_moment = m[:, -1] + M_tip

    # Form and scale residual
    res =  np.hstack([
        res_pos[:, :-1].flatten(order='F'),
        res_directors[:, :-1].flatten(order='F'),
        res_tip_force, res_tip_moment
    ])

    # res[:3*N] /= ds                    # position errors
    # res[3*N:6*N] /= 1.0                # rotation logmap 
    # res[-6:-3] /= (E * A)              # tip force balance
    # res[-3:] /= (E * I)                # tip moment balance

    return res

def residual_wrapper(q, params):
    return _residual(q, params)

# Simulate a rod===============================================================================================================
# Discretization
L = 0.5
N = 50
ds = L / N  # rest length of each segment

# Geometry
radius = 0.01  # m
A = np.pi * radius**2                 # m^2 Cross-sectional area
I = (np.pi / 4) * radius**4           # m^4 Second moment of area
J = 2 * I                             # m^2 Polar moment of area

# Material Properties
# Soft Silicone
E = 1e5       # Pa
G = 3.3e4     # Pa

# External forces and torques
f_ext = 1.0 * np.ones([3, N])
c_ext = 0 * np.ones([3, N])

F_tip = np.array([0, 0, 0])
M_tip = np.array([0, 0, 0])

# Boundary conditions
# Fixed base, free tip
r_base = np.zeros(3)
Theta_base = np.zeros(3)

rod_params = {
    "N": N,                                                  # Scalar
    "ds": ds,                                                # Scalar
    "f_ext": f_ext,                                          # (3, N)
    "c_ext": c_ext,                                          # (3, N)
    "r_base": r_base,                                        # (3,)
    "Theta_base": Theta_base,                                # (3,)
    "SB_inv": np.linalg.inv(np.diag([G*A, G*A, E*A])),       # (3, 3)
    "BB_inv": np.linalg.inv(np.diag([E*I, E*I, G*J])),       # (3, 3)
    "F_tip": F_tip,                                          # (3,)
    "M_tip": M_tip,                                           # (3,)
    "Area": A,
    "Young": E,
    "I": I
}


# The rod is intially straight along the Inertial Frame z axis
n_guess = 0.1 * np.ones(3).reshape(3, -1)                                                # (3, )
tau_guess = 0.0 * np.ones(3).reshape(3, -1)                                              # (3, )
r_guess = np.vstack([0.1 * np.ones((2, N)),                               
                    np.linspace(0, ds * N, N+1)[1:].reshape(1, -1)])     # (3, N)

# Bending in Y
r_guess = np.vstack([
    np.linspace(0, ds * N, N+1)[1:] * 0.1,   # bend slightly in x
    np.linspace(0, ds * N, N+1)[1:] * 0.5,   # bend slightly in y
    np.linspace(0, ds * N, N+1)[1:]          # z up
])

tau_guess = 1.0 * np.array([0, 1, 0]).reshape(3, -1)
Theta_guess = np.tile(np.array([0, 0.2, 0]), (N, 1)).T  # small guessial bend in y                                       # (3, N)

# Stack into guessial guess q
q_guess = np.hstack([n_guess, tau_guess, 
                    r_guess, 
                    Theta_guess]).flatten(order = 'F')        

# sol = root(
#     residual_wrapper,
#     q_guess,
#     args=(rod_params,),
#     method='hybr',
#     tol=1e-6
# )

sol = least_squares(
    residual_wrapper,     
    q_guess,             
    args=(rod_params,),   
    method='trf',         # Trust Region Reflective
    ftol=1e-5,            # Cost reduction (Col 4)
    xtol=1e-5,            # Step Norm (Col 5)
    gtol=1e-4,            # Optimality (Col 6)
    verbose=2             # Print detailed output for convergence tracking
)

# Check for convergence
threshold = N * ds / 20  # negligible values
if sol.success:
    print("Converged:", sol.message)
    q_sol = sol.x
    q_sol[np.abs(q_sol) < threshold] = 0.0

    n1, tau1, rD_sol, ThetaD_sol = _unpack(q_sol)
    r_sol = np.hstack([r_base.reshape(3, -1), rD_sol])
    Theta_sol = np.hstack([Theta_base.reshape(3, -1) ,ThetaD_sol])
else:
    raise RuntimeError(f"Root-finding failed: {sol.message}")

def axisEqual3D(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the rod as a line through the positions
ax.plot(r_sol[0, :], r_sol[1, :], r_sol[2, :], '-o', label='Rod Centerline')

# Axis labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Static Cosserat Rod Shape')
ax.legend()
axisEqual3D(ax)

plt.show()
