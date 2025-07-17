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

// -------- Matrix Tools --------
void mat_mult_4x4(double A[4][4], double B[4][4], double out[4][4]) {
    double tmp[4][4] = {0};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
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
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R[i][j] = (i == j);
        return;
    }

    double k_hat[3][3];
    double k_sq[3][3] = {0};
    hat_map(kappa, k_hat);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                k_sq[i][j] += k_hat[i][k] * k_hat[k][j];

    double A = sin(theta) / theta;
    double B = (1 - cos(theta)) / (theta * theta);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = (i == j) + A * k_hat[i][j] + B * k_sq[i][j];
}

void form_SE3(double R[3][3], double d[3], double g[4][4]) {
    memset(g, 0, sizeof(double) * 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            g[i][j] = R[i][j];
        g[i][3] = d[i];
    }
    g[3][3] = 1.0;
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
        double g[4][4] = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        };
        for (int i = 0; i < NUM_TRANSFORMS; ++i) {
            double R[3][3], g_next[4][4];
            double k[3] = {kappa[0][i], kappa[1][i], kappa[2][i]};
            double dd[3] = {d[0][i], d[1][i], d[2][i]};
            expmap_to_rot(k, R);
            form_SE3(R, dd, g_next);
            mat_mult_4x4(g, g_next, g);
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

    printf("==============================================\n");
    printf("SE(3) Benchmark with %dx timeit:\n", TIMEIT_N);
    printf("Total Time:          %.3f microseconds\n", total_time_ns / 1e3);
    printf("Average Time/Run:    %.3f microseconds\n", avg_time_ns / 1e3);
    printf("Peak Memory Usage:    %.2f KB\n", mem_kb);
    printf("==============================================\n");
    return 0;
}
