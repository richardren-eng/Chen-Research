#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define N_JOINTS 6
#define EPSILON 1e-12

// ---------- Helpers ----------

void hat_map(const double v[3], double out[3][3]) {
    out[0][0] = 0;      out[0][1] = -v[2]; out[0][2] = v[1];
    out[1][0] = v[2];   out[1][1] = 0;     out[1][2] = -v[0];
    out[2][0] = -v[1];  out[2][1] = v[0];  out[2][2] = 0;
}

void form_SE3(const double R[3][3], const double d[3], double g[4][4]) {
    memset(g, 0, sizeof(double)*16);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j)
            g[i][j] = R[i][j];
        g[i][3] = d[i];
    }
    g[3][3] = 1.0;
}

void expmap_so3(const double u[3], double R[3][3]) {
    double u_norm = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (u_norm < EPSILON) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R[i][j] = (i == j);
        return;
    }

    double u_hat[3][3], u_hat_sq[3][3] = {{0}};
    hat_map(u, u_hat);

    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            for (int k=0; k<3; ++k)
                u_hat_sq[i][j] += u_hat[i][k] * u_hat[k][j];

    double A = sin(u_norm) / u_norm;
    double B = (1 - cos(u_norm)) / (u_norm * u_norm);

    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            R[i][j] = (i==j) + A*u_hat[i][j] + B*u_hat_sq[i][j];
}

void adjoint_se3(const double g[4][4], double Ad[6][6]) {
    double R[3][3], p[3], p_hat[3][3], pR[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {
        p[i] = g[i][3];
        for (int j = 0; j < 3; ++j)
            R[i][j] = g[i][j];
    }
    hat_map(p, p_hat);
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            for (int k=0; k<3; ++k)
                pR[i][j] += p_hat[i][k] * R[k][j];

    memset(Ad, 0, sizeof(double)*36);
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) {
            Ad[i][j] = R[i][j];
            Ad[i+3][j] = pR[i][j];
            Ad[i+3][j+3] = R[i][j];
        }
}

void expmap_se3(const double xi[6], double theta, double g[4][4]) {
    double w[3] = {xi[0], xi[1], xi[2]}, v[3] = {xi[3], xi[4], xi[5]};
    double w_norm_sq = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    double R[3][3], d[3] = {0};

    if (w_norm_sq < EPSILON) {
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
                R[i][j] = (i == j);
        for (int i=0; i<3; ++i)
            d[i] = v[i] * theta;
    } else {
        double w_theta[3] = {w[0]*theta, w[1]*theta, w[2]*theta};
        double w_hat[3][3], w_hat_sq[3][3] = {{0}}, J[3][3];
        hat_map(w, w_hat);
        expmap_so3(w_theta, R);

        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
                for (int k=0; k<3; ++k)
                    w_hat_sq[i][j] += w_hat[i][k] * w_hat[k][j];

        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j) {
                double I = (i == j);
                J[i][j] = (I - R[i][j]) * w_hat[i][j] + (w_hat_sq[i][j] + I) * theta;
            }

        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
                d[i] += J[i][j] * v[j];
    }

    form_SE3(R, d, g);
}

// ---------- Example Main (FK + Jacobian) ----------

void run_fk_and_jacobian(const double xi[N_JOINTS][6], const double tet[N_JOINTS], const double tetdot[N_JOINTS], const double g_home[4][4], double g_st[4][4], double twist[6]) {
    double g[N_JOINTS][4][4];
    double g_chain[N_JOINTS][4][4];
    double J[N_JOINTS][6];

    for (int i = 0; i < N_JOINTS; ++i)
        expmap_se3(xi[i], tet[i], g[i]);

    memcpy(g_chain[0], g[0], sizeof(double)*16);
    for (int i = 1; i < N_JOINTS; ++i) {
        // g_chain[i] = g_chain[i-1] @ g[i]
        double tmp[4][4] = {{0}};
        for (int r=0; r<4; ++r)
            for (int c=0; c<4; ++c)
                for (int k=0; k<4; ++k)
                    tmp[r][c] += g_chain[i-1][r][k] * g[i][k][c];
        memcpy(g_chain[i], tmp, sizeof(double)*16);
    }

    // Final pose
    double tmp[4][4] = {{0}};
    for (int r=0; r<4; ++r)
        for (int c=0; c<4; ++c)
            for (int k=0; k<4; ++k)
                tmp[r][c] += g_chain[N_JOINTS-1][r][k] * g_home[k][c];
    memcpy(g_st, tmp, sizeof(double)*16);

    // Jacobian
    for (int j=0; j<6; ++j) J[0][j] = xi[0][j];
    for (int i = 1; i < N_JOINTS; ++i) {
        double Ad[6][6];
        adjoint_se3(g_chain[i-1], Ad);
        for (int j = 0; j < 6; ++j) {
            J[i][j] = 0;
            for (int k = 0; k < 6; ++k)
                J[i][j] += Ad[j][k] * xi[i][k];
        }
    }

    // End-effector twist
    for (int i=0; i<6; ++i) {
        twist[i] = 0;
        for (int j=0; j<N_JOINTS; ++j)
            twist[i] += J[j][i] * tetdot[j];
    }
}

int main() {
    // Constants
    const double L0 = 5.0, L1 = 2.0, L2 = 1.0;

    // Joint angles and velocities
    double tet[N_JOINTS] = {
        M_PI / 10, -M_PI / 6, -M_PI / 5, M_PI / 6, M_PI / 3, -M_PI
    };
    double tetdot[N_JOINTS] = {
        M_PI / 13, -M_PI / 12, M_PI / 7, M_PI / 10, -M_PI / 7, M_PI / 6
    };

    // Screw axes
    double omega[N_JOINTS][3] = {
        {  0,  0,  1 },
        { -1,  0,  0 },
        { -1,  0,  0 },
        {  0,  0,  1 },
        { -1,  0,  0 },
        {  0,  1,  0 }
    };
    double p[N_JOINTS][3] = {
        { 0, 0, 0 },
        { 0, 0, L0 },
        { 0, L1, L0 },
        { 0, L1 + L2, L0 },
        { 0, L1 + L2, L0 },
        { 0, L1 + L2, L0 }
    };

    // Compute xi = [omega, v]
    double xi[N_JOINTS][6];
    for (int i = 0; i < N_JOINTS; ++i) {
        xi[i][0] = omega[i][0];
        xi[i][1] = omega[i][1];
        xi[i][2] = omega[i][2];
        // v = -omega × p
        xi[i][3] = -(omega[i][1] * p[i][2] - omega[i][2] * p[i][1]);
        xi[i][4] = -(omega[i][2] * p[i][0] - omega[i][0] * p[i][2]);
        xi[i][5] = -(omega[i][0] * p[i][1] - omega[i][1] * p[i][0]);
    }

    // g_home
    double g_home[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, L1 + L2},
        {0, 0, 1, L0},
        {0, 0, 0, 1}
    };

    // Output
    double g_st[4][4];
    double twist[6];

    // Timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run FK and Jacobian
    run_fk_and_jacobian(xi, tet, tetdot, g_home, g_st, twist);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_sec = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("SE(3) pipeline time: %.9f us\n",  1000 * 1000 * elapsed_sec);

    // Optional: print end-effector pose and twist
    //printf("End-effector pose (g_st):\\n");
    //for (int i = 0; i < 4; ++i)
        //printf("%.4f %.4f %.4f %.4f\\n", g_st[i][0], g_st[i][1], g_st[i][2], g_st[i][3]);

    //printf("End-effector twist (spatial):\\n");
    //for (int i = 0; i < 6; ++i)
        //printf(" %.6f", twist[i]);
    //printf("\\n");

    return 0;
}
