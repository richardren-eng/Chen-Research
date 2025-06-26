import numpy as np
from scipy.linalg import expm
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

def _batch_norm(x):
    '''
    Calculate the norm of every column vector, which are stacked row-wise.

    Args:
        x = [x0, x1, x2, ..., x_{n-1}]:                    array containing column vectors stacked rowsie (m, n) np.array

    Return:
        [norm(x0), norm(x1), norm(x2), ..., norm(x_{n-1})] array containing norms of the column vectors (1, n) np.array
    '''
    return np.linalg.norm(x, axis = 0)

def _director_array_difference(RIB):
    '''
    Compute the relative rotation between an array of sequential rotation matrices.

    Args:
        RBI = [RB0_I, RB1_I, ..., RB{N-1}_I]:             (3, 3N) np.array where every 3x3 block going down row-wise is RBi_I.   
    
    Return:
        Theta = [Theta0, Theta1, ..., Theta_{N-2}]:   (3, N-1) np.array where every RBi_I = expm(hat(Theta_i))
    '''
    _, total_cols = RIB.shape
    N = total_cols // 3 - 1  # because (N+1) blocks

    Theta = np.zeros((3, N))

    for i in range(N):
        R_i = RIB[:, 3*i:3*(i+1)]
        R_ip1 = RIB[:, 3*(i+1):3*(i+2)]

        D = R_ip1 @ R_i.T
        trace_D = np.trace(D)
        cos_theta = (trace_D - 1) / 2

        # Numerical safety clamp for acos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if np.isclose(theta, 0):
            Theta[:, i] = np.zeros(3)
        else:
            axis_skew = D - D.T
            Theta[:, i] = (theta / (2 * np.sin(theta))) * _vee(axis_skew)

    return Theta

# Calculus Functions
def _array_difference(r):
    '''
    Computes first-order finite differences between columns of a matrix. Used for position.

    Args:
        r = [r0, r1, r2, ..., r_N]:              (m, N+1) np.array — where each column is a vector r_i

    Returns:
        [r1 - r0, r2 - r1, ..., r_N - r_{N-1}]:  (m, N) np.array — differences r[:,1:] - r[:,:-1]
    '''
    return r[:, 1:] - r[:, :-1]

def _bounded_array_difference(x):
    '''
    Constructs [x0, x1 - x0, x2 - x1, ..., x_{N-1} - x_{N-2}, -x_{n-1}]
    
    Args:
        x: (m, n) np.array — sequence of vectors [x0, x1, ..., x_{n-1}]
    
    Returns:
        Deltah_x: (m, n+1) np.array
    '''
    m, n = x.shape
    diff = x[:, 1:] - x[:, :-1]        # (m, n-1)
    first = x[:, [0]]                 # (m, 1)
    last = -x[:, [-1]]                # (m, 1)
    return np.hstack([first, diff, last])  # (m, n+1)

def _trapezoidal_quadrature(x):
    '''
    Constructs Ah_x = [x0, x1 - x0, x2 - x1, ..., x_{N-1} - x_{N-2}, -x_{n-1}]
    
    Arg:
        x: (m, n) np.array — sequence of vectors [x0, x1, ..., x_{n-1}]

    Return:
        Ah_x: (m, n+1) np.array

    '''
    left = x[:, [0]] / 2                     # (m, 1)
    middle = (x[:, :-1] + x[:, 1:]) / 2      # (m, n-1)
    right = x[:, [-1]] / 2                   # (m, 1)
    return np.hstack([left, middle, right])  # (m, n+1)

# Geometry Functions
def _calculate_edge_lengths(r):
    '''
    Calculate the edge lengths from the collection of all vertex points

    Arg:
        r: (3, N+1) np.array — sequence of position vectors resolved in the inertial frame [r0, r1, ..., r_N]
    
    Return:
        l: (1, N)  np.array -- scalar edge lengths 
    '''
    return _batch_norm(_array_difference(r))
    
def _calculate_tangent_unit_vectors_I(r):
    '''
    Calculate the tangent unit vectors of every edge resolved in the inertial frame.

    Arg:
        r: (3, N+1) np.array — sequence of position vectors resolved in the inertial frame [r0, r1, ..., r_N]
    '''
    return _array_difference(r) / _calculate_edge_lengths(r)

def _calculate_dilatation(l, lrest):
    '''
    Compute dilatations: 
    e_i = l_i / lrest_i -- arclength dilatation
    Eps_i = D_i / Drest_i -- Voronoi domain dilatation
    
    Args: 
        array of edge lengths:      (1, N) np.array
        array of rest edge lengths: (1, N) np.array

    Return:
        array of dilatations (1, N) np.array
    '''
    return l / lrest

def _calculate_voronoi_length(l):
    '''
    Calcuate Voronoi domain from edge lengths.

    Arg:
        l:          (1, N) np.array

    Return:
        voronoi_l:  (1, N-1) np.array
    '''
    return (l[1:] + l[:-1]) / 2

# Shear-Stretch strain and stress
def _calculate_shear_stretch_strains(RBI, arclength_dilatations, tangent_vectors_I):
    '''
    Compute the shear-stretch strain vectors for every edge.

    Args:
        RBI:                        (3, 3N) np.array
        arclength_dilatations:      (1, N) np.array
        tangent_vectors_I:          (3, N) np.array
    
    Return:
        sigma_B:                    (3, N) np.array
    '''
    N = arclength_dilatations.size

    # Body Frame b3 unit direction resolved in the body frame
    b3_B = np.array([0, 0, 1])

    # Compute the shear-stretch strain at every edge as sigma_B_i = e_i * RBi_I * tangent_i - b3_B
    sigma_B = np.zeros((3, N))
    for i in range(N):
        RBi_I = RBI[:, 3*i : 3*(i+1)]  # Extract 3x3 rotation matrix
        sigma_B[:, i] = arclength_dilatations[i] * (RBi_I @ tangent_vectors_I[:, i]) - b3_B
    
    return sigma_B

def _calculate_internal_stresses(SST_B, sigma_B, sigma_rest_B):
    '''
    Computes the shear-stretch stresses for every edge.

    Args:
        SST_B:                 (3, 3) np.array -- shear-stretch-tensor resolved in the body frame
        sigma_B:               (3, N) np.array 
        sigma_rest_B:          (3, N) np.array
    
        Return:
            internal_stress_B: (3, N) np.array
    '''
    _, N = sigma_B.shape
    internal_stress_B = np.zeros((3, N))

    # Compute the internal stress at edge_i as internal_stress_B_i = SST_B * (sigma_B_i - sigma_rest_B_i)
    for i in range(N):
        internal_stress_B[:, i] = SST_B @ (sigma_B[:, i] - sigma_rest_B[:, i])
    
    return internal_stress_B

# Bend-Twist strain and couple
def _calculate_curvatures_B(RBI, rest_voronoi_lengths):
    '''
    Compute the curvatures resolved in the body frame.

    Args:
        RBI: (3, 3N) np.array
        rest_voronoi_lengths: (1, N - 1) np.array

    Return:
        kappa_B: (3, N-1) np.array        
    '''
    N = rest_voronoi_lengths.size

    # Compute relative rotation vectors from each RBi_I to RB{i+1}_I
    Theta = _director_array_difference(RBI)     # (3, N-1)
    
    return Theta / rest_voronoi_lengths

def _calculate_internal_couples(BBT_B, kappa_B, kappa_rest_B):
    '''
    Compute the internal bending-twist couples for every edge

    Args:
        BBT_B:              (3, 3) np.array -- bending-twisting-tensor resolved in the body frame
        kappa_B:            (3, N-1)
        kappa_rest_B:       (3, N-1)

    Return:
        internal_couples_B: (3, N-1)
    '''
    _, Nminus1 = kappa_B.shape
    internal_couples_B = np.zeros((3, Nminus1))

    # Compute the internal couple at edge_i as internal_couple_i = BBT_B * (kappa_B_i - kappa_rest_B_i)
    for i in range(Nminus1):
        internal_couples_B[:, i] = BBT_B @ (kappa_B[:, i] - kappa_rest_B[:, i])
    
    return internal_couples_B

# Internal force and torque
def _static_internal_force_balance_I(arclength_dilatation, RBI, internal_stresses_B):
    '''
    Computes Deltah(internal force) resolved in the Inertial Frame for each edge

    Args:
        arclength_dilatation (1, N) 
        RBI:                 (3, 3N)
        internal_stresses:   (3, N)

    Return:
        Deltah(internal_forces): (3, N)
    '''
    N = arclength_dilatation.size
    n_I = np.zeros((3, N))

    # Compute the internal force at edge_i as n_I_i = transpose(RBi_I) * internal_stresses_B_i / arclength_dilatation_i
    for i in range(N):
        RBi_I = RBI[:, 3*i : 3*(i+1)]  # Extract 3x3 rotation matrix
        n_I[:, i] = RBi_I.T @ internal_stresses_B[:, i] / arclength_dilatation[i]
    
    return _bounded_array_difference(n_I) # This plus the external forces resolved in the inertial frame equals 0

def _static_internal_torque_balance_B(rest_lengths, tangents_I, 
                                RBI,
                              rest_voronoi_lengths, voronoi_dilatations, 
                              curvatures_B, 
                              internal_stresses_B, internal_couples_B):
    '''
    Computes the quantity where quantity + torque_external_B = 0 for the static case.

    Args:
        rest_lengths:           (1, N)
        tangents_I:             (3, N)
        RBI:                    (3, 3N)
        rest_voronoi_lengths:   (1, N-1)
        voronoi_dilatations:    (1, N-1)
        internal_stresses_B:      (3, N)
        internal_couples_B:     (3, N-1)
        curvatures_B:           (3, N-1)

    Return:
        The thing where (thing + external moments in the body frame = 0):     (3, N)
    '''
    N = rest_lengths.size
    
    # Bend-twist 2D torque
    Deltah_bend_twist_2D = _bounded_array_difference(internal_couples_B / voronoi_dilatations**3) # (3, N)
    
    # Bend-twist 3D torque
    kappa_cross_internal_couple = np.zeros((3, N-1))
    for i in range(N-1):
        kappa_cross_internal_couple[:, i] = np.linalg.cross(curvatures_B[:, i], internal_couples_B[:, i])
    
    factor = (rest_voronoi_lengths / voronoi_dilatations**3)  # (1, N-1)
    Ah_bend_twist_3D = _trapezoidal_quadrature(kappa_cross_internal_couple * factor) # (3, N)

    # Shear-stretch coupled torque
    shear_stretch_coupled = np.zeros((3, N))
    for i in range(N):
        RBi_I = RBI[:, 3*i : 3*(i+1)]  # Extract 3x3 rotation matrix
        shear_stretch_coupled[:, i] = np.linalg.cross(RBi_I @ tangents_I[:, i] , internal_stresses_B[:, i]) * rest_lengths[i]
    
    return Deltah_bend_twist_2D + Ah_bend_twist_3D + shear_stretch_coupled # This plus the external torque resolved in the body frame equals zero

def _unpack_static_residual(q):
    '''
    Extract the position and director collections from the residual.

    Arg:
        q: (3, 2N) -- residual vector where q = [r, Theta]

    Returns:
        r:   (3, N + 1)
        Theta: (3, N - 1) -- where RB{i+1}_I = expm(hat(Theta_i)) * RBi_I
    '''
    N = int(q.shape[1]/2)
    r = q[:, 0:N+1]
    Theta = q[:, N+1:]

    return r, Theta

def _compute_static_residual(q, rod_params):
    '''
    Build the residual for a static cosserat rod.
    '''
    # Unpack the state
    r, Theta = _unpack_static_residual(q)
    N = int(r.shape[1] - 1)

    # Reconstruct the directors
    RBI = np.zeros((3, 3*N))
    RBI[:, 0:3] = rod_params["RB0I"] # Base rotation matrix

    for i in range(N - 1):
        Theta_hat = _hat(Theta[:, i])
        RBI[:, 3*(i+1):3*(i+2)] = expm(Theta_hat) @ RBI[:, 3*i:3*(i+1)]

    # Geometry Calculations
    l = _calculate_edge_lengths(r)
    voronoi_length = _calculate_voronoi_length(l)
    e = _calculate_dilatation(l, rod_params["rest_lengths"])
    tangents_I = _calculate_tangent_unit_vectors_I(r)

    # Strain Calculations
    sigma_B = _calculate_shear_stretch_strains(RBI, e, tangents_I)
    kappa_B = _calculate_curvatures_B(RBI, rod_params["rest_voronoi_lengths"])

    # Stress Calculations
    internal_stress_B = _calculate_internal_stresses(
        rod_params["SST_B"], sigma_B, rod_params["sigma_rest_B"]
    )
    internal_couple_B = _calculate_internal_couples(
        rod_params["BBT_B"], kappa_B, rod_params["kappa_rest_B"]
    )

    # Calculate the internal force and torque balance terms
    internal_force_balance_I = _static_internal_force_balance_I(e, RBI, internal_stress_B)
    internal_torque_balance_B = _static_internal_torque_balance_B(
        rod_params["rest_lengths"], tangents_I, RBI,
        rod_params["rest_voronoi_lengths"],
        _calculate_dilatation(voronoi_length, rod_params["rest_voronoi_lengths"]),
        kappa_B, 
        internal_stress_B, internal_couple_B
    )

    # External Loads
    f_ext_I = rod_params["external_forces_I"]
    tau_ext_B = rod_params["external_torques_B"]

    # Residual (should equal zero)
    return np.hstack([internal_force_balance_I + f_ext_I,       # (3, N+1)
                      internal_torque_balance_B + tau_ext_B,    # (3, N)
                      ])                                        # TOTAL SHAPE: (3, 2N + 1)


def residual_wrapper(q_flat, rod_params):
    q = q_flat.reshape(3, -1)  # reshape back to (3, 2N)
    res = _compute_static_residual(q, rod_params)
    return res.flatten()  # root expects 1D output


# Simulate a rod===============================================================================================================
# Discretization
N = 50
ds = 0.01  # rest length of each segment

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
    "rest_voronoi_lengths": _calculate_voronoi_length(np.ones(N) * ds),
    "sigma_rest_B": np.zeros((3, N)),
    "kappa_rest_B": np.zeros((3, N - 1)),
    "SST_B": np.diag([G*A, G*A, E*A]),
    "BBT_B": np.diag([E*I, E*I, G*J]),
    "external_forces_I": np.zeros((3, N+1)),
    "external_torques_B": np.zeros((3, N)),
    "RB0I": np.eye(3)
}

Fy_tip = 0.01
rod_params["external_forces_I"][:, -10] = np.array([0.0, Fy_tip, 0.0])

# The rod is intially straight along the Inertial Frame z axis
r_init = np.linspace(0, ds * N, N+1).reshape(1, -1)
r_init = np.vstack([np.zeros((2, N+1)), r_init])  # (3, N+1)
Theta_init = np.zeros((3, N-1))

# Stack into initial guess q
q_init = np.hstack([r_init, Theta_init])  # shape (3, 2N)
q_init = np.hstack([q_init, np.zeros((3,1))]) # Shape (3, 2N + 1)

# Solve using root-finder
sol = root(
    residual_wrapper,
    q_init.flatten(),
    args=(rod_params,),
    method='hybr',
    tol=1e-6
)

# Check for convergence
threshold = ds/10  # negligible values
if sol.success:
    print("Converged:", sol.message)
    q_sol = sol.x.reshape(3, -1)
    q_sol[np.abs(q_sol) < threshold] = 0.0

    r_sol, Theta_sol = _unpack_static_residual(q_sol)
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

