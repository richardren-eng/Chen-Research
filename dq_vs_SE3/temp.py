import numpy as np
import timeit
from lib.clifford_and_SE3 import *
from lib.benchmark_funcs import measure_peak_memory


# SE(3) ===============================================================================================================
L0 = 5
L1 = 2
L2 = 1

tet1 = np.pi / 10
tet2 = -np.pi / 6
tet3 = -np.pi / 5
tet4 = np.pi / 6
tet5 = np.pi / 3
tet6 = -np.pi 

tetdot1 = np.pi / 13
tetdot2 = -np.pi / 12
tetdot3 = np.pi / 7
tetdot4 = np.pi / 10
tetdot5 = -np.pi / 7
tetdot6 = np.pi / 6

omega1 = np.array([0, 0, 1])
omega2 = np.array([-1, 0, 0])
omega3 = np.array([-1, 0, 0])
omega4 = np.array([0, 0, 1])
omega5 = np.array([-1, 0, 0])
omega6 = np.array([0, 1, 0])

p1 = np.array([0, 0, 0])
p2 = np.array([0, 0, L0])
p3 = np.array([0, L1, L0])
p4 = np.array([0, L1 + L2, L0])
p5 = np.array([0, L1 + L2, L0])
p6 = np.array([0, L1 + L2, L0])

N = 6
omega = np.stack([globals()[f"omega{i+1}"] for i in range(N)])                 # (N, 3)
p     = np.stack([globals()[f"p{i+1}"]     for i in range(N)])                 # (N, 3)
v = -np.cross(omega, p)                                                        # (N, 3)


tet = np.array([tet1, tet2, tet3, tet4, tet5, tet6])
tetdot = np.array([tetdot1, tetdot2, tetdot3, tetdot4, tetdot5, tetdot6])

g_home = np.array([[1, 0, 0, 0],
                   [0, 1, 0, L1 + L2],
                   [0, 0, 1, L0],
                   [0, 0, 0, 1]])

Q_home = np.array([1, 0, 0, 0,   0, 0, 0.5 * (L1 + L2), 0.5 * L0])




def se3_pipeline():
    xi = np.hstack([omega, v])                                                     # (N, 6)
    g = np.stack([expmap_se3(xi[i], tet[i]) for i in range(xi.shape[0])])          # (N, 4, 4)

    # Forward Kinematics
    # Stores [g1, g1*g2, g1*g2*g3, ..., g1*g2*g3*...*g_N] where N = number of joints
    g_chains = np.zeros((N, 4, 4))   
    g_chains[0] = g[0]
    for i in range(1, N): g_chains[i] = g_chains[i-1] @ g[i]

    # Compute pose of the end-effector
    g_st = g_chains[-1] @ g_home    # (4, 4)

    # Compute Spatial Manipulator Jacobian
    # [xi1, xi2', xi3', ..., xiN']
    J_st_S = np.zeros((N, 6))
    J_st_S[0] = xi[0]
    for i in range(1, N): J_st_S[i] = Ad(g_chains[i-1]) @ xi[i]

    # Compute End-effector Twist
    twist_st_S = J_st_S @ tetdot 
    



def dq_pipeline():
    zerosNx1 = np.zeros((N, 1))
    xi_dq = np.hstack([zerosNx1, omega, zerosNx1, v])                              # (N, 8) xi_dq = (0, ω) + ε (0, v), stack v and omega opposite of SE(3).
    Q = np.stack([expmap_twistDQ(xi_dq[i], tet[i]) for i in range(N)])      # (N, 8)

    # Forward Kinematics
    # Stores [Q1, Q1⨂Q2, Q1⨂Q2⨂Q3, ..., Q1⨂Q2⨂Q3⨂...⨂Q_N] where N = number of joints
    Q_chains = np.zeros((N, 8))   
    Q_chains[0] = Q[0]
    for i in range(1, N): Q_chains[i] = dualquatmult(Q_chains[i-1], Q[i])

    # Compute pose of the end-effector
    Q_st = dualquatmult(Q_chains[-1], Q_home)   # (8, )

    # Compute the DQ Spatial Manipulator Jacobian
    J_st_S_fat = np.zeros((N, 8))
    J_st_S_fat[0] = xi_dq[0]
    for i in range(1, N): J_st_S_fat[i] = dualquatsandwich(Q_chains[i-1], xi_dq[i])

    # Extract the 6 x N Spatial Manipulator Jacobian
    J_st_S_fromdq = np.delete(J_st_S_fat, [0, 4], axis=1)

    # Compute End-effector Twist
    twist_st_S_fromdq = J_st_S_fromdq @ tetdot 



if __name__ == "__main__":
    # abstol = 1e-8

    # # Ensure Product of Exponentials results are the same
    # assert np.linalg.norm(dualquat_to_SE3(Q_st) - g_st, ord='fro') < abstol
    # print("FORWARD KINEMATICS PASSED") 

    # # Ensure Spatial Manipualtor Jacobian results are the same
    # # Since SE3 stacks xi = [v; omega] and DQ stacks xi_dq = [(0, omega); (0, v)], columns 1-N/2 and N/2-N should be swapped in order to compare.
    # assert np.linalg.norm(J_st_S_fromdq - J_st_S, ord='fro') < abstol 
    # assert np.linalg.norm(twist_st_S - twist_st_S_fromdq) < abstol
    # print("SPATIAL MANIPULATOR JACOBIAN PASSED") 

    # ================== TIMEIT =====================
    num_timeit_runs = 1000
    run_dq_first = 1

    if run_dq_first:
        dq_time = timeit.timeit(dq_pipeline, number=num_timeit_runs)
        peak_mem_dq = measure_peak_memory(dq_pipeline)
        se3_time = timeit.timeit(se3_pipeline, number=num_timeit_runs)
        peak_mem_se3 = measure_peak_memory(se3_pipeline)
        se3_avg = se3_time / num_timeit_runs
        dq_avg = dq_time / num_timeit_runs    
    else:
        se3_time = timeit.timeit(se3_pipeline, number=num_timeit_runs)
        peak_mem_se3 = measure_peak_memory(se3_pipeline)
        dq_time = timeit.timeit(dq_pipeline, number=num_timeit_runs)
        peak_mem_dq = measure_peak_memory(dq_pipeline)
        se3_avg = se3_time / num_timeit_runs
        dq_avg = dq_time / num_timeit_runs

    print("="*50)
    print("         POE and Spatial Jacobian Benchmark Results         ")
    print("="*50)
    print(f"{'Order of Testing:':<35} {'DQ, SE(3)' if run_dq_first else 'SE(3), DQ'}")
    print(f"{'Number of timeit runs:':<35} {num_timeit_runs}")
    print(f"{'SE(3) Avg Time:':<35} {se3_avg * 1000:.2f} ms")
    print(f"{'Dual Quaternion Avg Time:':<35} {dq_avg * 1000:.2f} ms")
    print(f"{'DQ is how many × faster than SE3:':<35} {se3_avg / dq_avg:.2f}× faster")
    print(f"{'Peak Memory (SE3):':<40} {peak_mem_se3:,.1f} KB")
    print(f"{'Peak Memory (Dual Quaternion):':<40} {peak_mem_dq:,.1f} KB")
    print(f"{'Memory saving (DQ vs SE3):':<40} {peak_mem_se3 / peak_mem_dq:,.2f}× less memory")
    print("="*50)