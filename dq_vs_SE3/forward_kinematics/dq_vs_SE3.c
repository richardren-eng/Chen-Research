#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define NUM_TRANSFORMS 100
#define PI 3.141592653589793

// -------- Timing --------
double time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

// -------- Matrix Tools --------
void mat_mult_4x4(double A[4][4], double B[4][4], double out[4][4]) {
    double tmp[4][4] = {0};
    for (int i=0; i<4; ++i)
        for (int j=0; j<4; ++j)
            for (int k=0; k<4; ++k)
                tmp[i][j] += A[i][k] * B[k][j];
    memcpy(out, tmp, sizeof(tmp));
}

void hat_map(double v[3], double out[3][3]) {
    out[0][0] = 0;      out[0][1] = -v[2]; out[0][2] = v[1];
    out[1][0] = v[2];   out[1][1] = 0;     out[1][2] = -v[0];
    out[2][0] = -v[1];  out[2][1] = v[0];  out[2][2] = 0;
}

void expmap_to_rot(double kappa[3], double R[3][3]) {
    double theta = sqrt(kappa[0]*kappa[0] + kappa[1]*kappa[1] + kappa[2]*kappa[2]);
    if (theta < 1e-8) {
        // Identity
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
                R[i][j] = (i == j);
        return;
    }

    double k_hat[3][3];
    double k_sq[3][3] = {0};
    hat_map(kappa, k_hat);

    // k_hat squared
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            for (int k=0; k<3; ++k)
                k_sq[i][j] += k_hat[i][k] * k_hat[k][j];

    double A = sin(theta)/theta;
    double B = (1 - cos(theta)) / (theta*theta);

    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j)
            R[i][j] = (i==j) + A * k_hat[i][j] + B * k_sq[i][j];
}

void form_SE3(double R[3][3], double d[3], double g[4][4]) {
    memset(g, 0, sizeof(double)*16);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j)
            g[i][j] = R[i][j];
        g[i][3] = d[i];
    }
    g[3][3] = 1.0;
}

// -------- Dual Quaternion Tools --------
void form_dual_quat(double kappa[3], double d[3], double Q[8]) {
    double theta = sqrt(kappa[0]*kappa[0] + kappa[1]*kappa[1] + kappa[2]*kappa[2]);
    double half_theta = 0.5 * theta;

    double cos_ht = cos(half_theta);
    double sin_ht = (theta < 1e-8) ? 0 : sin(half_theta)/theta;

    double rx = kappa[0] * sin_ht;
    double ry = kappa[1] * sin_ht;
    double rz = kappa[2] * sin_ht;

    Q[0] = cos_ht;
    Q[1] = rx;
    Q[2] = ry;
    Q[3] = rz;

    // Dual part
    Q[4] = -0.5 * (d[0]*rx + d[1]*ry + d[2]*rz);
    Q[5] = 0.5 * (cos_ht*d[0] + d[1]*rz - d[2]*ry);
    Q[6] = 0.5 * (cos_ht*d[1] + d[2]*rx - d[0]*rz);
    Q[7] = 0.5 * (cos_ht*d[2] + d[0]*ry - d[1]*rx);
}

void dual_quat_multiply(double A[8], double B[8], double out[8]) {
    double ar=A[0], ax=A[1], ay=A[2], az=A[3];
    double ad0=A[4], ad1=A[5], ad2=A[6], ad3=A[7];
    double br=B[0], bx=B[1], by=B[2], bz=B[3];
    double bd0=B[4], bd1=B[5], bd2=B[6], bd3=B[7];

    out[0] = ar*br - ax*bx - ay*by - az*bz;
    out[1] = ar*bx + ax*br + ay*bz - az*by;
    out[2] = ar*by - ax*bz + ay*br + az*bx;
    out[3] = ar*bz + ax*by - ay*bx + az*br;

    out[4] = ar*bd0 - ax*bd1 - ay*bd2 - az*bd3 + ad0*br - ad1*bx - ad2*by - ad3*bz;
    out[5] = ar*bd1 + ax*bd0 + ay*bd3 - az*bd2 + ad0*bx + ad1*br + ad2*bz - ad3*by;
    out[6] = ar*bd2 - ax*bd3 + ay*bd0 + az*bd1 + ad0*by - ad1*bz + ad2*br + ad3*bx;
    out[7] = ar*bd3 + ax*bd2 - ay*bd1 + az*bd0 + ad0*bz + ad1*by - ad2*bx + ad3*br;
}



// -------- Main Benchmark --------
int main() {
    srand(807);
    double kappa[3][NUM_TRANSFORMS];
    double d[3][NUM_TRANSFORMS];

    for (int i=0; i<NUM_TRANSFORMS; ++i)
        for (int j=0; j<3; ++j) {
            kappa[j][i] = ((double)rand()) / RAND_MAX;
            d[j][i] = ((double)rand()) / RAND_MAX;
        }

    struct timespec t1, t2;
    
    flush_cache();
    // --- Dual Quaternion Benchmark ---
    double Q[8] = {1,0,0,0, 0,0,0,0};
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < NUM_TRANSFORMS; ++i) {
        double k2[3] = {kappa[0][i], kappa[1][i], kappa[2][i]};
        double dd2[3] = {d[0][i], d[1][i], d[2][i]};
        double Qi[8];
        form_dual_quat(k2, dd2, Qi);
        double Qout[8];
        dual_quat_multiply(Q, Qi, Qout);
        memcpy(Q, Qout, sizeof(Qout));
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double dq_time = time_diff(t1, t2);
        
    flush_cache();
    // --- SE(3) Benchmark ---
    double g[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < NUM_TRANSFORMS; ++i) {
        double R[3][3], g_next[4][4];
        double k[3] = {kappa[0][i], kappa[1][i], kappa[2][i]};
        double dd[3] = {d[0][i], d[1][i], d[2][i]};
        expmap_to_rot(k, R);
        form_SE3(R, dd, g_next);
        mat_mult_4x4(g, g_next, g);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double se3_time = time_diff(t1, t2);

    // ----- Convert Dual Quaternion to SE(3) to check match-----
    double qr[4] = {Q[0], Q[1], Q[2], Q[3]};
    double qd[4] = {Q[4], Q[5], Q[6], Q[7]};

    // Compute conjugate of qr
    double qr_conj[4] = {qr[0], -qr[1], -qr[2], -qr[3]};

    // Hamilton product qd * qr_conj
    double t_quat[4];
    t_quat[0] = qd[0]*qr_conj[0] - qd[1]*qr_conj[1] - qd[2]*qr_conj[2] - qd[3]*qr_conj[3];
    t_quat[1] = qd[0]*qr_conj[1] + qd[1]*qr_conj[0] + qd[2]*qr_conj[3] - qd[3]*qr_conj[2];
    t_quat[2] = qd[0]*qr_conj[2] - qd[1]*qr_conj[3] + qd[2]*qr_conj[0] + qd[3]*qr_conj[1];
    t_quat[3] = qd[0]*qr_conj[3] + qd[1]*qr_conj[2] - qd[2]*qr_conj[1] + qd[3]*qr_conj[0];

    // Translation vector = 2 * vector part of above
    double t[3] = {2 * t_quat[1], 2 * t_quat[2], 2 * t_quat[3]};

    // Rotation matrix from qr
    double w = qr[0], x = qr[1], y = qr[2], z = qr[3];
    double R[3][3] = {
        {1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)},
        {2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)},
        {2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)}
    };

    // Build SE(3) from DQ
    double gDQ[4][4] = {{0}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            gDQ[i][j] = R[i][j];
        gDQ[i][3] = t[i];
    }
    gDQ[3][3] = 1.0;

    // ----- Compare Frobenius norm -----
    double err = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            err += (g[i][j] - gDQ[i][j]) * (g[i][j] - gDQ[i][j]);

    err = sqrt(err);

    printf("Frobenius norm between SE(3) and DQ result: %.12f\n", err);
    if (err > 1e-8) {
        fprintf(stderr, "ERROR: DQ and SE(3) mismatch!\n");
        return 1;
    }
    printf("Transformation match passed (error < 1e-8)\n");



    // --- Output ---
    printf("==============================================\n");
    printf("SE(3) Time:            %.6f ms\n", se3_time);
    printf("Dual Quaternion Time: %.6f ms\n", dq_time);
    printf("DQ Speedup:               %.2f× faster\n", se3_time / dq_time);
    printf("==============================================\n");
    return 0;
}
