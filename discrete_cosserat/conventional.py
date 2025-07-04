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

# Calculus Functions
def _array_difference(r):
    '''
    Computes first-order finite differences between columns of a matrix. Used for position.

    Args:
        r = [r1, r2, ..., r_{N+1}]:               (m, N+1) np.array — where each column is a vector r_i

    Returns:
        [r2 - r1, r3 - r2, ..., r_{N + 1} - r_N]:  (m, N) np.array 
    '''
    return r[:, 1:] - r[:, :-1]

def _voronoi_domain(l):
    '''
    Calcuate Voronoi quantity.

    Arg:
        l = [l1, l2, l3, ...,l_{N-1}, l_N]                                      (1, N) np.array

    Return:
        voronoi_l = [(l1 + l2) / 2, (l2 + l3) / 2, ..., (l_{N-1} + l_N) / 2]:   (1, N-1) np.array 
    '''
    return (l[1:] + l[:-1]) / 2

def _Deltah(tau):
    '''
    Constructs the discrete finite difference definition of tau: Deltah_tau = [tau1, tau2 - tau1, tau3 - tau2, ..., tau_{N-1} - tau_{N-2}, -tau_{N-1}]

    Arg:
        tau: (m, N-1)


        Return:
            Deltah_tau: (m, N) 
    '''
    return np.hstack([tau[:, 0].reshape(-1, 1), 
                      tau[:, 1:] - tau[:, :-1],
                        -tau[:, -1].reshape(-1, 1)])

def _Ah(n):
    '''
    Constructs the trapezoidal quadrature of n: Ah_n = [n1 / 2, (n2 + n1) / 2, (n3 + n2) / 2, ..., (n_{N-1} + n_N) / 2]
    
    Arg:
        n: (m, N) np.array — sequence of vectors [n1, n2, n3,..., n_{N-1}, n_N]

    Return:
        Ah_n: (m, N+1) np.array

    '''
    left = n[:, [0]] / 2                     # (m, 1)
    middle = (n[:, :-1] + n[:, 1:]) / 2      # (m, N-1)
    right = n[:, [-1]] / 2                   # (m, 1)
    return np.hstack([left, middle, right])  # (m, N+1)

def _unpack_residual2(q, N):
    ''''
    Extract n1, tau1, r and PhiBI from the flattened residual array and reshape then to a collection of column vectors

    Args: 
        q = [n1, tau1, | r2, r3, ..., r_{N+1}, | Phi_{B1_2} Phi_{B3_I}, ..., Phi_{BN_I}]      
        Need to reshape r to (3, N) and PhiBU to (3, N-1)
    
    Returns:
        r, PhiBI         (3, N) and (3, N-1) respectively
    '''
    n1 = q[0:3]
    tau1 = q[3:6]
    rD = q[6: 6+3*N].reshape(3, -1, order='F')
    PhiBID = q[6+3*N:].reshape(3, -1, order='F')

    return n1, tau1, rD, PhiBID


def _residual(q, parameters):
    N = parameters["N"]
    rest_lens = parameters["rest_lengths"]   # (N, )
    rest_D = _voronoi_domain(rest_lens)      # (N, )
    SST_B = parameters["SST_B"]              # (3, 3)
    BBT_B = parameters["BBT_B"]              # (3, 3)
    F_ext = parameters["f_ext"]              # (3, N+1)
    tau_ext = parameters["tau_ext"]          # (3, N)
    
    n1, tau1, rD, PhiBID = _unpack_residual2(q, N)           # (3, N) , (3, N-1)
    r = np.hstack([np.zeros(3).reshape(3, -1), rD])          # (3, N+1)
    PhiBI = np.hstack([np.zeros(3).reshape(-3, 1), PhiBID])  # (3, N)

    # Calculate quantities available from r
    diff_r = _array_difference(r)            # displacements r_{i+1} - r_i                                        (3, N)
    lens = np.linalg.norm(diff_r, axis=0)    # lengths of every edge                                              (N, )
    tangents = diff_r / lens                 # tangent vectors on every edge resolved in the inertial frame       (3, N)
    e_dil = lens / rest_lens                 # arclength dilatations                                              (N, )
    D = _voronoi_domain(lens)                # Voronoi domains                                                    (N-1, )
    Eps = D / rest_D                         # Voronoi strains                                                    (N-1, )

    # Calculate rotation matrices 
    RBI = np.zeros([3, 3*N])              # (3, 3N)
    for i in range(N):
        RBI[:, 3*i : 3*(i+1)] = expm(_hat(PhiBI[:, i]))

    # Calculate bodyframe shear-stretch strains
    sigma_B = np.zeros([3, N])            # (3, N)
    for i in range(N):
        sigma_B[:, i] = e_dil[i] * RBI[:, 3*i : 3*(i+1)] @ tangents[:, i] - np.array([0, 0, 1])

    # Calculate bodyframe curvatures
    kappa_B = np.zeros([3, N-1])          # (3, N-1)
    for i in range(N-1):
        kappa_B[:, i] = 1 / rest_D[i] * _vee(logm(RBI[:, 3*(i+1) : 3*(i+2)] @ RBI[:, 3*i : 3*(i+1)].T))

    
    # Internal Force balance
    n = np.zeros([3, N])
    for i in range(N):
        n[:, i] = 1 / e_dil[i] * RBI[:, 3*i : 3*(i+1)].T @ SST_B @ sigma_B[:, i] 
    
    force_balance = _Deltah(n) + F_ext   # (3, N+1)
    force_balance[:, 0] = n1 + F_ext[:, 0]

    # Internal moment balance
    # Bending and Twisting Contribution
    bend_twist_2D = np.zeros([3, N-1])
    bend_twist_3D = np.zeros([3, N-1])
    for i in range(N-1):
        bend_twist_2D[:, i] = BBT_B @ kappa_B[:, i] / Eps[i]**3
        bend_twist_3D[:, i] = np.linalg.cross(kappa_B[:, i] , BBT_B @ kappa_B[:, i]) / Eps[i]**3 * D[i]
    
    # Shear and Stretch Internal Couple
    shear_stretch_couple = np.zeros([3, N])
    for i in range(N):
        shear_stretch_couple[:, i] = np.linalg.cross(RBI[:, 3*i : 3*(i+1)] @ tangents[:, i] , SST_B @ sigma_B[:, i])

    torque_balance = _Deltah(bend_twist_2D) + _Ah(bend_twist_3D) + shear_stretch_couple + tau_ext   # (3, N)
    torque_balance[:, 0] = tau1 + tau_ext[:, 0]

    residual = np.hstack([force_balance.flatten(order = 'F'), torque_balance.flatten(order = 'F')]) # 
    
    # Inside _residual
    debug_vars.update({
        "r": r,
        "PhiBI": PhiBI,
        "diff_r": diff_r,
        "lens": lens,
        "tangents": tangents,
        "e_dil": e_dil,
        "D": D,
        "Eps": Eps,
        "RBI": RBI,
        "sigma_B": sigma_B,
        "kappa_B": kappa_B,
        "n": n,
        "force_balance": force_balance,
        "bend_twist_2D": bend_twist_2D,
        "bend_twist_3D": bend_twist_3D,
        "shear_stretch_couple": shear_stretch_couple,
        "torque_balance": torque_balance,
        "res": residual,
    })
    
    return residual

def residual_wrapper(q, parameters):
    return _residual(q, parameters)

# Simulate a rod===============================================================================================================
# Discretization
L = 0.2
N = 20
ds = L / N  # rest length of each segment

# Geometry
radius = 0.001  # m
A = np.pi * radius**2                 # m^2 Cross-sectional area
I = (np.pi / 4) * radius**4           # m^4 Second moment of area
J = 2 * I                             # m^2 Polar moment of area

# Material Properties
# Soft Silicone
E = 1e5       # Pa
G = 3.3e4     # Pa

# External forces and torques
f_external_I = np.zeros([3, N+1])
f_external_I[:, -1] = np.array([0, 10, 0])

rod_params = {
    "N": N,
    "rest_lengths": np.ones(N) * ds,
    "rest_D": _voronoi_domain(np.ones(N) * ds),
    "sigma_rest_B": np.zeros((3, N)),
    "kappa_rest_B": np.zeros((3, N - 1)),
    "SST_B": np.diag([G*A, G*A, E*A]),
    "BBT_B": np.diag([E*I, E*I, G*J]),
    "f_ext": f_external_I,
    "tau_ext": np.zeros((3, N))
}


# The rod is intially straight along the Inertial Frame z axis
n_init = 0.0 * np.ones(3).reshape(3, -1)
tau_init = 0.0 * np.ones(3).reshape(3, -1)
r_init = np.vstack([0.0 * np.ones((2, N)), 
                    np.linspace(0, ds * N, N+1)[1:].reshape(1, -1)])     # (3, N+1)
PhiBI_init = 0.0 * np.ones((3, N-1))

# Stack into initial guess q
q_init = np.hstack([n_init, tau_init, r_init, PhiBI_init]).flatten(order = 'F')        

# Solve using root-finder
# sol = root(
#     residual_wrapper,
#     q_init.flatten(),
#     args=(rod_params,),
#     method='hybr',
#     tol=1e-4,
# )

sol = least_squares(
    residual_wrapper,     
    q_init,             
    args=(rod_params,),   
    method='trf',         # Trust Region Reflective
    ftol=1e-5,            # Cost reduction (Col 4)
    xtol=1e-5,            # Step Norm (Col 5)
    gtol=1e-5,            # Optimality (Col 6)
    verbose=2             # Print detailed output for convergence tracking
)

# Check for convergence
threshold = ds / 10  # negligible values
if sol.success:
    print("Converged:", sol.message)
    q_sol = sol.x
    q_sol[np.abs(q_sol) < threshold] = 0.0

    n1, tau1, rD_sol, ThetaD_sol = _unpack_residual2(q_sol, N)
    r_sol = np.hstack([np.zeros(3).reshape(3, -1), rD_sol])
    Theta_sol = np.hstack([np.zeros(3).reshape(3, -1) ,ThetaD_sol])
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

# Equal scaling
x_range = r_sol[0, :]
y_range = r_sol[1, :]
z_range = r_sol[2, :]

x_min, x_max = np.min(x_range), np.max(x_range)
y_min, y_max = np.min(y_range), np.max(y_range)
z_min, z_max = np.min(z_range), np.max(z_range)

max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
x_mid = (x_max + x_min) / 2.0
y_mid = (y_max + y_min) / 2.0
z_mid = (z_max + z_min) / 2.0

ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)
axisEqual3D(ax)

plt.show()
































# Validate helper functions
# np.random.seed(50)
# rrest = np.array([
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5],
#     [1, 2, 3, 4, 5]
# ])

# r = np.array([
#     [1, 2, 3, 4, 5],
#     [2, 3, 6, 7, 8],
#     [3, 5, 7, 8, 10]
# ])

# diff_r = _array_difference(r)
# norm_r = _batch_norm(r)

# Deltah_r = _bounded_array_difference(r)
# Ah_r = _trapezoidal_quadrature(r)

# restedge_lens = _calculate_edge_lengths(rrest)
# edge_lens = _calculate_edge_lengths(r)
# tangents = _calculate_tangent_unit_vectors_I(r)

# arclength_dilatations = _calculate_dilatation(edge_lens, restedge_lens)

# voronoi_domain = _calculate_voronoi_length(edge_lens)
# rest_voronoi_domain = _calculate_voronoi_length(restedge_lens)

# voronoi_dilatations = _calculate_dilatation(voronoi_domain, rest_voronoi_domain)

# # Generate an array of rotation matrices
# om0 = np.random.rand(3, )
# om1 = np.random.rand(3, )
# om2 = np.random.rand(3, )
# om3 = np.random.rand(3, )
# om4 = np.random.rand(3, )

# RB0_I = expm(_hat(om0))
# RB1_I = expm(_hat(om1))
# RB2_I = expm(_hat(om2))
# RB3_I = expm(_hat(om3))
# RB4_I = expm(_hat(om4))
# RBI = np.hstack([RB0_I, RB1_I, RB2_I, RB3_I, RB4_I])

# # Calculate the rotation vectors
# Theta = _director_array_difference(RBI)
# Theta0 = Theta[:, 0]
# Theta1 = Theta[:, 1]
# Theta2 = Theta[:, 2]
# Theta3 = Theta[:, 3]

# # Ensure the rotation matrices computed from the rotation vectors match the truths
# RB1_I_calc = expm(_hat(Theta0)) @ RB0_I
# RB2_I_calc = expm(_hat(Theta1)) @ RB1_I_calc
# RB3_I_calc = expm(_hat(Theta2)) @ RB2_I_calc
# RB4_I_calc = expm(_hat(Theta3)) @ RB3_I_calc
# RBI_calc = np.hstack([RB0_I, RB1_I_calc, RB2_I_calc, RB3_I_calc, RB4_I_calc])
