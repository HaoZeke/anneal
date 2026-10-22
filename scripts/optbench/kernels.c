/* Native twins of optbench's Python kernels, for the anneal runner.
 *
 * Each function is the arithmetic of the optbench kernel it names, in the
 * same order: optbench.potentials.lennard_jones, binary_lennard_jones and
 * optbench.tip4p.tip4p. They return the energy and write the gradient (not
 * the force). anneal_globalopt.py checks each against the Python kernel at
 * the start of a run and at every hit, so a disagreement stops the row.
 */
#include <math.h>
#include <stddef.h>

int anneal_lj(const double *x, int atoms, double cutoff, double *energy_out, double *grad) {
    double energy = 0.0;
    double cutoff2 = cutoff * cutoff;
    for (int i = 0; i < 3 * atoms; i++) {
        grad[i] = 0.0;
    }
    for (int i = 0; i < atoms; i++) {
        for (int j = i + 1; j < atoms; j++) {
            double dx = x[3 * i] - x[3 * j];
            double dy = x[3 * i + 1] - x[3 * j + 1];
            double dz = x[3 * i + 2] - x[3 * j + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0) || !(r2 < cutoff2)) {
                continue;
            }
            double inv_r2 = 1.0 / r2;
            double sr2 = inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            energy += sr12 - sr6;
            double slope = 24.0 * (sr6 - 2.0 * sr12) * inv_r2;
            grad[3 * i] += slope * dx;
            grad[3 * i + 1] += slope * dy;
            grad[3 * i + 2] += slope * dz;
            grad[3 * j] -= slope * dx;
            grad[3 * j + 1] -= slope * dy;
            grad[3 * j + 2] -= slope * dz;
        }
    }
    *energy_out = 4.0 * energy;
    return 0;
}

int anneal_binary_lj(const double *x, const long *species, int atoms, double cutoff,
                     double sigma_a, double sigma_b, double *energy_out, double *grad) {
    double energy = 0.0;
    double cutoff2 = cutoff * cutoff;
    for (int i = 0; i < 3 * atoms; i++) {
        grad[i] = 0.0;
    }
    for (int i = 0; i < atoms; i++) {
        double si = species[i] == 1 ? sigma_a : sigma_b;
        for (int j = i + 1; j < atoms; j++) {
            double dx = x[3 * i] - x[3 * j];
            double dy = x[3 * i + 1] - x[3 * j + 1];
            double dz = x[3 * i + 2] - x[3 * j + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0) || !(r2 < cutoff2)) {
                continue;
            }
            double sj = species[j] == 1 ? sigma_a : sigma_b;
            double sij = 0.5 * (si + sj);
            double inv_r2 = 1.0 / r2;
            double sr2 = (sij * sij) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            energy += sr12 - sr6;
            double slope = 24.0 * (sr6 - 2.0 * sr12) * inv_r2;
            grad[3 * i] += slope * dx;
            grad[3 * i + 1] += slope * dy;
            grad[3 * i + 2] += slope * dz;
            grad[3 * j] -= slope * dx;
            grad[3 * j + 1] -= slope * dy;
            grad[3 * j + 2] -= slope * dz;
        }
    }
    *energy_out = 4.0 * energy;
    return 0;
}

#define TIP4P_C6 2552.24
#define TIP4P_C12 2510.4e3
#define TIP4P_COULOMB 1389.354848

static const double CHARGES[4] = {0.0, 0.52, 0.52, -1.04};

/* GMIN rotation of the four reference sites; d[s][a][k] is d r_sa / d omega_k. */
static void rotate(const double *omega, const double ref[4][3], double out[4][3], double d[4][3][3]) {
    double alpha2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    double alpha = sqrt(alpha2);
    int small = alpha < 1e-8;
    double cosine, c3, sine_over, d_cosine, d_c3, d_sine, d_alpha[3];
    if (small) {
        cosine = 1.0;
        c3 = -0.5;
        sine_over = 1.0;
        d_cosine = d_c3 = d_sine = 0.0;
        d_alpha[0] = d_alpha[1] = d_alpha[2] = 0.0;
    } else {
        cosine = cos(alpha);
        double sine = sin(alpha);
        c3 = (cosine - 1.0) / alpha2;
        sine_over = sine / alpha;
        d_cosine = -sine;
        d_c3 = (-sine * alpha2 - (cosine - 1.0) * 2.0 * alpha) / (alpha2 * alpha2);
        d_sine = (cosine * alpha - sine) / alpha2;
        for (int k = 0; k < 3; k++) {
            d_alpha[k] = omega[k] / alpha;
        }
    }
    for (int s = 0; s < 4; s++) {
        const double *p = ref[s];
        double projection = p[0] * omega[0] + p[1] * omega[1] + p[2] * omega[2];
        double crossed[3] = {
            omega[1] * p[2] - omega[2] * p[1],
            omega[2] * p[0] - omega[0] * p[2],
            omega[0] * p[1] - omega[1] * p[0],
        };
        for (int a = 0; a < 3; a++) {
            out[s][a] = cosine * p[a] - c3 * projection * omega[a] - sine_over * crossed[a];
        }
        for (int k = 0; k < 3; k++) {
            double unit[3] = {0.0, 0.0, 0.0};
            unit[k] = 1.0;
            double sc = small ? 0.0 : d_cosine * d_alpha[k];
            double s3 = small ? 0.0 : d_c3 * d_alpha[k];
            double ss = small ? 0.0 : d_sine * d_alpha[k];
            double ucross[3] = {
                unit[1] * p[2] - unit[2] * p[1],
                unit[2] * p[0] - unit[0] * p[2],
                unit[0] * p[1] - unit[1] * p[0],
            };
            for (int a = 0; a < 3; a++) {
                double term = sc * p[a];
                term = term - s3 * projection * omega[a];
                term = term - c3 * p[k] * omega[a];
                term = term - c3 * projection * unit[a];
                term = term - ss * crossed[a];
                term = term - sine_over * ucross[a];
                d[s][a][k] = term;
            }
        }
    }
}

/* x is n blocks of cx, cy, cz, lx, ly, lz; ref is O, H, H, M in the centre-of-mass
 * frame; work holds 4 * 3 * n + 4 * 3 * 3 * n + 4 * 3 * n doubles. */
int anneal_tip4p(const double *x, int molecules, const double *ref_flat, double *work,
                 double *energy_out, double *grad) {
    double ref[4][3];
    for (int s = 0; s < 4; s++) {
        for (int a = 0; a < 3; a++) {
            ref[s][a] = ref_flat[3 * s + a];
        }
    }
    double (*placed)[4][3] = (double (*)[4][3])work;
    double (*deriv)[4][3][3] = (double (*)[4][3][3])(work + 12 * (size_t)molecules);
    double (*site)[4][3] = (double (*)[4][3])(work + 48 * (size_t)molecules);
    for (int m = 0; m < molecules; m++) {
        double rotated[4][3];
        rotate(x + 6 * m + 3, ref, rotated, deriv[m]);
        for (int s = 0; s < 4; s++) {
            for (int a = 0; a < 3; a++) {
                placed[m][s][a] = x[6 * m + a] + rotated[s][a];
                site[m][s][a] = 0.0;
            }
        }
    }
    double energy = 0.0;
    for (int l = 0; l < molecules - 1; l++) {
        for (int r = l + 1; r < molecules; r++) {
            double delta[3];
            for (int a = 0; a < 3; a++) {
                delta[a] = placed[l][0][a] - placed[r][0][a];
            }
            double distance2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            double inverse = 1.0 / distance2;
            double r6 = inverse * inverse * inverse;
            double r12 = r6 * r6;
            energy += TIP4P_C12 * r12 - TIP4P_C6 * r6;
            double slope = (-12.0 * TIP4P_C12 * r12 + 6.0 * TIP4P_C6 * r6) * inverse;
            for (int a = 0; a < 3; a++) {
                site[l][0][a] += slope * delta[a];
                site[r][0][a] -= slope * delta[a];
            }
            for (int sa = 1; sa < 4; sa++) {
                for (int sb = 1; sb < 4; sb++) {
                    for (int a = 0; a < 3; a++) {
                        delta[a] = placed[l][sa][a] - placed[r][sb][a];
                    }
                    distance2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                    double distance = sqrt(distance2);
                    double pair = TIP4P_COULOMB * CHARGES[sa] * CHARGES[sb] / distance;
                    energy += pair;
                    slope = -pair / distance2;
                    for (int a = 0; a < 3; a++) {
                        site[l][sa][a] += slope * delta[a];
                        site[r][sb][a] -= slope * delta[a];
                    }
                }
            }
        }
    }
    for (int m = 0; m < molecules; m++) {
        for (int a = 0; a < 3; a++) {
            double total = 0.0;
            for (int s = 0; s < 4; s++) {
                total += site[m][s][a];
            }
            grad[6 * m + a] = total;
        }
        for (int k = 0; k < 3; k++) {
            double total = 0.0;
            for (int s = 0; s < 4; s++) {
                for (int a = 0; a < 3; a++) {
                    total += deriv[m][s][a][k] * site[m][s][a];
                }
            }
            grad[6 * m + 3 + k] = total;
        }
    }
    *energy_out = energy;
    return 0;
}
