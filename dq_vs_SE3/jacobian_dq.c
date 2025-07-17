#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define N 6
#define EPSILON 1e-12

void dualquatmult(const double Q1[8], const double Q2[8], double Qout[8]) {
    double rw1 = Q1[0], rx1 = Q1[1], ry1 = Q1[2], rz1 = Q1[3];
    double tw1 = Q1[4], tx1 = Q1[5], ty1 = Q1[6], tz1 = Q1[7];
    double rw2 = Q2[0], rx2 = Q2[1], ry2 = Q2[2], rz2 = Q2[3];
    double tw2 = Q2[4], tx2 = Q2[5], ty2 = Q2[6], tz2 = Q2[7];

    Qout[0] = rw1*rw2 - rx1*rx2 - ry1*ry2 - rz1*rz2;
    Qout[1] = rw1*rx2 + rx1*rw2 + ry1*rz2 - rz1*ry2;
    Qout[2] = rw1*ry2 - rx1*rz2 + ry1*rw2 + rz1*rx2;
    Qout[3] = rw1*rz2 + rx1*ry2 - ry1*rx2 + rz1*rw2;

    Qout[4] = rw1*tw2 - rx1*tx2 - ry1*ty2 - rz1*tz2 + tw1*rw2 - tx1*rx2 - ty1*ry2 - tz1*rz2;
    Qout[5] = rw1*tx2 + rx1*tw2 + ry1*tz2 - rz1*ty2 + tw1*rx2 + tx1*rw2 + ty1*rz2 - tz1*ry2;
    Qout[6] = rw1*ty2 - rx1*tz2 + ry1*tw2 + rz1*tx2 + tw1*ry2 - tx1*rz2 + ty1*rw2 + tz1*rx2;
    Qout[7] = rw1*tz2 + rx1*ty2 - ry1*tx2 + rz1*tw2 + tw1*rz2 + tx1*ry2 - ty1*rx2 + tz1*rw2;
}

void dualquatconj(const double Q[8], double Qc[8]) {
    Qc[0] =  Q[0];  Qc[1] = -Q[1];  Qc[2] = -Q[2];  Qc[3] = -Q[3];
    Qc[4] =  Q[4];  Qc[5] = -Q[5];  Qc[6] = -Q[6];  Qc[7] = -Q[7];
}

void dualquatsandwich(const double Q[8], const double xi[8], double out[8]) {
    double tmp[8], Qc[8];
    dualquatmult(Q, xi, tmp);
    dualquatconj(Q, Qc);
    dualquatmult(tmp, Qc, out);
}

void expmap_twistDQ(const double xi_dq[8], double theta, double out[8]) {
    double w[3] = {xi_dq[1], xi_dq[2], xi_dq[3]};
    double v[3] = {xi_dq[5], xi_dq[6], xi_dq[7]};
    double w_norm = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

    if (w_norm < EPSILON) {
        out[0] = 1.0; out[1] = 0.0; out[2] = 0.0; out[3] = 0.0;
        out[4] = 0.0; out[5] = 0.5 * v[0] * theta;
        out[6] = 0.5 * v[1] * theta;
        out[7] = 0.5 * v[2] * theta;
    } else {
        double half_theta = 0.5 * theta;
        double sin_half = sin(half_theta);
        double cos_half = cos(half_theta);
        out[0] = cos_half;
        out[1] = w[0] * sin_half;
        out[2] = w[1] * sin_half;
        out[3] = w[2] * sin_half;
        out[4] = 0.0;
        out[5] = sin_half * v[0];
        out[6] = sin_half * v[1];
        out[7] = sin_half * v[2];
    }
}

int main() {
    const double L0 = 5, L1 = 2, L2 = 1;
    double tet[N] = {M_PI/10, -M_PI/6, -M_PI/5, M_PI/6, M_PI/3, -M_PI};
    double tetdot[N] = {M_PI/13, -M_PI/12, M_PI/7, M_PI/10, -M_PI/7, M_PI/6};
    double omega[N][3] = {
        {0,0,1}, {-1,0,0}, {-1,0,0}, {0,0,1}, {-1,0,0}, {0,1,0}
    };
    double p[N][3] = {
        {0,0,0}, {0,0,L0}, {0,L1,L0}, {0,L1+L2,L0}, {0,L1+L2,L0}, {0,L1+L2,L0}
    };

    double xi_dq[N][8];
    for (int i = 0; i < N; ++i) {
        double vx = -(omega[i][1]*p[i][2] - omega[i][2]*p[i][1]);
        double vy = -(omega[i][2]*p[i][0] - omega[i][0]*p[i][2]);
        double vz = -(omega[i][0]*p[i][1] - omega[i][1]*p[i][0]);
        xi_dq[i][0] = 0;
        xi_dq[i][1] = omega[i][0];
        xi_dq[i][2] = omega[i][1];
        xi_dq[i][3] = omega[i][2];
        xi_dq[i][4] = 0;
        xi_dq[i][5] = vx;
        xi_dq[i][6] = vy;
        xi_dq[i][7] = vz;
    }

    double Q[N][8], Q_chain[N][8], J_fat[N][8], Q_home[8] = {1, 0, 0, 0, 0, 0, 0.5*(L1+L2), 0.5*L0};

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; ++i)
        expmap_twistDQ(xi_dq[i], tet[i], Q[i]);

    memcpy(Q_chain[0], Q[0], sizeof(double)*8);
    for (int i = 1; i < N; ++i)
        dualquatmult(Q_chain[i-1], Q[i], Q_chain[i]);

    double Q_st[8];
    dualquatmult(Q_chain[N-1], Q_home, Q_st);

    memcpy(J_fat[0], xi_dq[0], sizeof(double)*8);
    for (int i = 1; i < N; ++i)
        dualquatsandwich(Q_chain[i-1], xi_dq[i], J_fat[i]);

    double J_st[6][N];
    for (int i = 0; i < N; ++i) {
        J_st[0][i] = J_fat[i][1];
        J_st[1][i] = J_fat[i][2];
        J_st[2][i] = J_fat[i][3];
        J_st[3][i] = J_fat[i][5];
        J_st[4][i] = J_fat[i][6];
        J_st[5][i] = J_fat[i][7];
    }

    double twist[6] = {0};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < N; ++j)
            twist[i] += J_st[i][j] * tetdot[j];

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Dual Quaternion pipeline time: %.9f us\n", 1000 * 1000 * elapsed);

    return 0;
}
