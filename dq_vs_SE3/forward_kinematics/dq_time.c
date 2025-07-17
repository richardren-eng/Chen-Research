#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/resource.h>


#define NUM_TRANSFORMS 1000
#define TIMEIT_N 10000
#define PI 3.141592653589793

// -------- Accurate Timing (CPU Time) --------
double time_diff_ns(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
}

// -------- Dual Quaternion Tools --------
void form_dual_quat(double kappa[3], double d[3], double Q[8]) {
    double theta = sqrt(kappa[0]*kappa[0] + kappa[1]*kappa[1] + kappa[2]*kappa[2]);
    double half_theta = 0.5 * theta;

    double cos_ht = cos(half_theta);
    double sin_ht = (theta < 1e-8) ? 0 : sin(half_theta) / theta;

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

    for (int i = 0; i < NUM_TRANSFORMS; ++i)
        for (int j = 0; j < 3; ++j) {
            kappa[j][i] = ((double)rand()) / RAND_MAX;
            d[j][i] = ((double)rand()) / RAND_MAX;
        }


    struct timespec t1, t2;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

    for (int repeat = 0; repeat < TIMEIT_N; ++repeat) {
        double Q[8] = {1, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < NUM_TRANSFORMS; ++i) {
            double k2[3] = {kappa[0][i], kappa[1][i], kappa[2][i]};
            double dd2[3] = {d[0][i], d[1][i], d[2][i]};
            double Qi[8];
            form_dual_quat(k2, dd2, Qi);
            double Qout[8];
            dual_quat_multiply(Q, Qi, Qout);
            memcpy(Q, Qout, sizeof(Qout));
        }
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
    double total_time_ns = time_diff_ns(t1, t2);
    double avg_time_ns = total_time_ns / TIMEIT_N;

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    #ifdef __APPLE__
        // macOS returns bytes
        double mem_kb = usage.ru_maxrss / 1024.0;
    #else
        // Linux returns kilobytes
        double mem_kb = usage.ru_maxrss;
    #endif

    // --- Output ---
    printf("==============================================\n");
    printf("Dual Quaternion Benchmark with %dx timeit:\n", TIMEIT_N);
    printf("Total CPU Time:       %.3f microseconds\n", total_time_ns / 1e3);
    printf("Average Time/Run:     %.3f microseconds\n", avg_time_ns / 1e3);
    printf("Peak Memory Usage:    %.2f KB\n", mem_kb);
    printf("==============================================\n");

    return 0;
}
