import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import root
from matplotlib import pyplot as plt

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


def _unpack_residual(q, N):
    ''''
    Extract r and PhiBI from the flattened residual array and reshape then to a collection of column vectors

    Args: 
        q = [[r1, r2, ..., r_{N+1}].flattened, [Phi_{B1_I}, Phi_{B2_I}, ..., Phi_{BN_I}].flattend] shape (6*N + 3, )
        There are N + 1 vertices hence N + 1 r values
        There are N material frames hence N Phi values, where expm(hat(Phi_{Bi_I})) = RBi_I
        Need to reshape r to (3, N+1) and PhiBU to (3, N)
    
    Returns:
        r, PhiBI         (3, N+1) and (3, N) respectively
    '''
    return q[:3*(N+1)].reshape(3,-1,order='F') , q[3*(N+1):].reshape(3, -1, order = 'F') 


def _residual(q, parameters):
    N = parameters["N"]
    rest_lens = parameters["rest_lengths"]   # (N, )
    rest_D = _voronoi_domain(rest_lens)      # (N, )
    SST_B = parameters["SST_B"]              # (3, 3)
    BBT_B = parameters["BBT_B"]              # (3, 3)
    F_ext = parameters["F_ext"]              # (3, N+1)
    tau_ext = parameters["tau_ext"]          # (3, N)
 
    r, PhiBI = _unpack_residual(q, N)

    # Calculate quantities available from r
    diff_r = _array_difference(r)         # displacements r_{i+1} - r_i                                        (3, N)
    lens = np.linalg.norm(diff_r, axis=0)        # lengths of every edge                                       (N, )
    tangents = diff_r / lens              # tangent vectors on every edge resolved in the inertial frame       (3, N)
    e_dil = lens / rest_lens              # arclength dilatations                                              (N, )
    D = _voronoi_domain(lens)             # Voronoi domains                                                    (N-1, )
    Eps = D / rest_D                      # Voronoi strains                                                    (N-1, )

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

    residual = np.hstack([force_balance.flatten(order = 'F'), torque_balance.flatten(order = 'F')]) # (6*N + 3, )
    return residual

def residual_wrapper(q, parameters):
    return _residual(q, parameters)

# Simulate a rod===============================================================================================================
# Discretization
N = 10
ds = 0.05  # rest length of each segment

# Geometry
radius = 0.001  # m
A = np.pi * radius**2                 # m^2 Cross-sectional area
I = (np.pi / 4) * radius**4           # m^4 Second moment of area
J = 2 * I                             # m^2 Polar moment of area

# Material Properties
# Soft Silicone
E = 1e5       # Pa
G = 3.3e4     # Pa

rod_params = {
    "N": N,
    "rest_lengths": np.ones(N) * ds,
    "rest_D": _voronoi_domain(np.ones(N) * ds),
    "sigma_rest_B": np.zeros((3, N)),
    "kappa_rest_B": np.zeros((3, N - 1)),
    "SST_B": np.diag([G*A, G*A, E*A]),
    "BBT_B": np.diag([E*I, E*I, G*J]),
    "F_ext": np.zeros((3, N+1)),
    "tau_ext": np.zeros((3, N))
}

Fy_tip = 0.01
rod_params["F_ext"][:, -1] = np.array([0.0, Fy_tip, 0.0])

# The rod is intially straight along the Inertial Frame z axis
r_init = np.linspace(0, ds * N, N+1).reshape(1, -1)
r_init = np.vstack([np.zeros((2, N+1)), r_init])  # (3, N+1)
PhiBI_init = np.zeros((3, N))

# Stack into initial guess q
q_init = np.hstack([r_init, PhiBI_init])  # shape (3, 2N)
q_init = q_init.flatten(order = 'F')

# Solve using root-finder
sol = root(
    residual_wrapper,
    q_init.flatten(),
    args=(rod_params,),
    method='hybr',
    tol=1e-6
)

# Check for convergence
threshold = ds / 10  # negligible values
if sol.success:
    print("Converged:", sol.message)
    q_sol = sol.x
    q_sol[np.abs(q_sol) < threshold] = 0.0

    r_sol, Theta_sol = _unpack_residual(q_sol, N)
else:
    raise RuntimeError(f"Root-finding failed: {sol.message}")


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

