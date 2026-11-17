// Periodic MBD@rsSCS. The dipole tensor follows the real-space plus
// Ewald split used by pymbd: screening is the short-range Gaussian
// dipole on lattice images, and the MBD Hamiltonian is the damped
// bare dipole plus the Ewald correction, averaged on a shifted
// Monkhorst-Pack mesh.

bool invert3(const double a[3][3], double inv[3][3]) {
  const double c0 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
  const double c1 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
  const double c2 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
  const double det = a[0][0] * c0 + a[0][1] * c1 + a[0][2] * c2;
  if (!(std::abs(det) > 1e-18)) {
    return false;
  }
  const double d = 1.0 / det;
  inv[0][0] = c0 * d;
  inv[1][0] = c1 * d;
  inv[2][0] = c2 * d;
  inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * d;
  inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * d;
  inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * d;
  inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * d;
  inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * d;
  inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * d;
  return true;
}

void image_range(const double latt[3][3], double radius, int out[3]) {
  double lt[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      lt[i][j] = latt[j][i];
    }
  }
  double inv[3][3];
  if (!invert3(lt, inv)) {
    throw std::invalid_argument("MBD cell is singular");
  }
  double rl[3][3];
  double nrm[3];
  for (int i = 0; i < 3; ++i) {
    double s = 0.0;
    for (int j = 0; j < 3; ++j) {
      rl[i][j] = 2.0 * kPi * inv[i][j];
      s += rl[i][j] * rl[i][j];
    }
    nrm[i] = std::sqrt(s);
  }
  for (int col = 0; col < 3; ++col) {
    double sep = 0.0;
    for (int row = 0; row < 3; ++row) {
      sep += latt[row][col] * (rl[row][col] / nrm[col]);
    }
    if (!(sep > 1e-12)) {
      throw std::invalid_argument("MBD cell has a vanishing layer spacing");
    }
    out[col] = static_cast<int>(std::ceil(radius / sep + 0.5));
    if (out[col] < 0) {
      out[col] = 0;
    }
  }
}

TB t_erfc_dip(double x, double y, double z, double alpha) {
  const double r2 = x * x + y * y + z * z;
  const double r = std::sqrt(r2);
  const double r3 = r2 * r;
  const double r5 = r3 * r2;
  const double ar = alpha * r;
  const double expa = std::exp(-ar * ar);
  const double span = 2.0 * ar / std::sqrt(kPi);
  const double B = (std::erfc(ar) + span * expa) / r3;
  const double C =
      (3.0 * std::erfc(ar) + span * (3.0 + 2.0 * ar * ar) * expa) / r5;
  const double rvec[3] = {x, y, z};
  TB t{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      t.t[a][b] = -C * rvec[a] * rvec[b];
      if (a == b) {
        t.t[a][b] += B;
      }
    }
  }
  return t;
}

struct DipoleWork {
  int n = 0;
  std::vector<double> re;
  std::vector<double> im;
  explicit DipoleWork(int n_)
      : n(n_), re(static_cast<std::size_t>(9 * n_ * n_), 0.0),
        im(static_cast<std::size_t>(9 * n_ * n_), 0.0) {}
  int at(int i, int a, int j, int b) const { return (3 * i + a) * (3 * n) + (3 * j + b); }
};

void add_tensor(DipoleWork *d, int i, int j, const TB &t, double pre, double pim) {
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const int idx = d->at(i, a, j, b);
      d->re[static_cast<std::size_t>(idx)] += t.t[a][b] * pre;
      d->im[static_cast<std::size_t>(idx)] += t.t[a][b] * pim;
    }
  }
}

// gg selects the short-range Gaussian dipole. Otherwise the tensor is
// the Fermi-damped bare dipole, plus the Ewald correction when asked.
void accumulate_dipole(DipoleWork *d, const double *pos, const std::vector<double> &alpha,
                       const std::vector<double> &rvdw, const double latt[3][3], double volume,
                       double beta, double a, bool gg, bool ewald, const double *kpt) {
  const int n = d->n;
  const double ewald_alpha = 2.5 / std::cbrt(volume);
  const double real_cut = 6.0 / ewald_alpha;
  int span[3];
  image_range(latt, real_cut, span);
  for (int iz = -span[2]; iz <= span[2]; ++iz) {
    for (int iy = -span[1]; iy <= span[1]; ++iy) {
      for (int ix = -span[0]; ix <= span[0]; ++ix) {
        const double rc[3] = {ix * latt[0][0] + iy * latt[1][0] + iz * latt[2][0],
                              ix * latt[0][1] + iy * latt[1][1] + iz * latt[2][1],
                              ix * latt[0][2] + iy * latt[1][2] + iz * latt[2][2]};
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            const double x = pos[3 * i] - pos[3 * j] + rc[0];
            const double y = pos[3 * i + 1] - pos[3 * j + 1] + rc[1];
            const double z = pos[3 * i + 2] - pos[3 * j + 2] + rc[2];
            const double r = std::sqrt(x * x + y * y + z * z);
            if (!(r > 1e-10)) {
              continue;
            }
            const double s_vdw = beta * (rvdw[static_cast<std::size_t>(i)] +
                                         rvdw[static_cast<std::size_t>(j)]);
            const double f = fermi(r, s_vdw, a);
            TB t{};
            if (gg) {
              const double ai = alpha[static_cast<std::size_t>(i)];
              const double aj = alpha[static_cast<std::size_t>(j)];
              const double si = std::cbrt(std::sqrt(2.0 / kPi) * ai / 3.0);
              const double sj = std::cbrt(std::sqrt(2.0 / kPi) * aj / 3.0);
              t = t_erf(x, y, z, std::sqrt(si * si + sj * sj));
              for (int u = 0; u < 3; ++u) {
                for (int v = 0; v < 3; ++v) {
                  t.t[u][v] *= (1.0 - f);
                }
              }
            } else {
              t = t_bare(x, y, z);
              const TB te = t_erfc_dip(x, y, z, ewald_alpha);
              const TB tb = t;
              for (int u = 0; u < 3; ++u) {
                for (int v = 0; v < 3; ++v) {
                  t.t[u][v] = f * tb.t[u][v] + te.t[u][v] - tb.t[u][v];
                }
              }
            }
            double pre = 1.0;
            double pim = 0.0;
            if (kpt != nullptr) {
              const double dot = kpt[0] * x + kpt[1] * y + kpt[2] * z;
              pre = std::cos(-dot);
              pim = std::sin(-dot);
            }
            add_tensor(d, i, j, t, pre, pim);
          }
        }
      }
    }
  }
  if (!ewald) {
    return;
  }
  double rlatt[3][3];
  {
    double lt[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lt[i][j] = latt[j][i];
      }
    }
    double inv[3][3];
    if (!invert3(lt, inv)) {
      throw std::invalid_argument("MBD reciprocal cell is singular");
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rlatt[i][j] = 2.0 * kPi * inv[i][j];
      }
    }
  }
  const double rec_cut = 10.0 * ewald_alpha;
  int gspan[3];
  image_range(rlatt, rec_cut, gspan);
  for (int iz = -gspan[2]; iz <= gspan[2]; ++iz) {
    for (int iy = -gspan[1]; iy <= gspan[1]; ++iy) {
      for (int ix = -gspan[0]; ix <= gspan[0]; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }
        double gvec[3] = {};
        const int idx[3] = {ix, iy, iz};
        for (int c = 0; c < 3; ++c) {
          gvec[c] = idx[0] * rlatt[0][c] + idx[1] * rlatt[1][c] + idx[2] * rlatt[2][c];
        }
        double ktot[3];
        for (int c = 0; c < 3; ++c) {
          ktot[c] = gvec[c] + (kpt != nullptr ? kpt[c] : 0.0);
        }
        const double k2 = ktot[0] * ktot[0] + ktot[1] * ktot[1] + ktot[2] * ktot[2];
        if (!(k2 > 0.0) || std::sqrt(k2) > rec_cut) {
          continue;
        }
        const double scale =
            4.0 * kPi / volume * std::exp(-k2 / (4.0 * ewald_alpha * ewald_alpha)) / k2;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            const double rx = pos[3 * i] - pos[3 * j];
            const double ry = pos[3 * i + 1] - pos[3 * j + 1];
            const double rz = pos[3 * i + 2] - pos[3 * j + 2];
            const double gdotr = gvec[0] * rx + gvec[1] * ry + gvec[2] * rz;
            const double pre = kpt != nullptr ? std::cos(gdotr) : std::cos(gdotr);
            const double pim = kpt != nullptr ? std::sin(gdotr) : 0.0;
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                const int id = d->at(i, a, j, b);
                const double kab = scale * ktot[a] * ktot[b];
                d->re[static_cast<std::size_t>(id)] += kab * pre;
                d->im[static_cast<std::size_t>(id)] += kab * pim;
              }
            }
          }
        }
      }
    }
  }
  const double self = -4.0 * ewald_alpha * ewald_alpha * ewald_alpha / (3.0 * std::sqrt(kPi));
  for (int i = 0; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      d->re[static_cast<std::size_t>(d->at(i, a, i, a))] += self;
    }
  }
  double ksq = 0.0;
  if (kpt != nullptr) {
    ksq = kpt[0] * kpt[0] + kpt[1] * kpt[1] + kpt[2] * kpt[2];
  }
  if (std::sqrt(ksq) > 1e-15 && kpt != nullptr) {
    const double scale =
        4.0 * kPi / volume * std::exp(-ksq / (4.0 * ewald_alpha * ewald_alpha)) / ksq;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            d->re[static_cast<std::size_t>(d->at(i, a, j, b))] += scale * kpt[a] * kpt[b];
          }
        }
      }
    }
  } else {
    const double g0 = 4.0 * kPi / (3.0 * volume);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int a = 0; a < 3; ++a) {
          d->re[static_cast<std::size_t>(d->at(i, a, j, a))] += g0;
        }
      }
    }
  }
}

double mbd_periodic_hartree(const double *pos_a, std::size_t n, const int *z, const double *box_a,
                            const std::vector<double> &h, const std::vector<double> *c6ratio,
                            const DispersionConfig &cfg) {
  if (box_a == nullptr) {
    throw std::invalid_argument("periodic MBD needs a cell");
  }
  for (int c = 0; c < 3; ++c) {
    if (cfg.mbd_k[c] < 1) {
      throw std::invalid_argument("MBD k-mesh must be at least 1");
    }
  }
  std::vector<double> pos(3 * n), boxv(9);
  for (std::size_t i = 0; i < pos.size(); ++i) {
    pos[i] = pos_a[i] / kBohr;
  }
  for (int i = 0; i < 9; ++i) {
    boxv[static_cast<std::size_t>(i)] = box_a[i] / kBohr;
  }
  double latt[3][3];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      latt[r][c] = boxv[static_cast<std::size_t>(3 * r + c)];
    }
  }
  const CellGeom cell = cell_from_box(boxv.data());
  const Scaled base = scale_ts(z, n, h, c6ratio);
  std::vector<double> alpha_w(static_cast<std::size_t>(kMbdNfreq) * n);
  for (int f = 0; f < kMbdNfreq; ++f) {
    std::vector<double> ad(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double omega = (4.0 / 3.0) * base.c6[i] / (base.alpha[i] * base.alpha[i]);
      const double u = kMbdFreq[f];
      ad[i] = base.alpha[i] / (1.0 + (u / omega) * (u / omega));
    }
    DipoleWork dip(static_cast<int>(n));
    accumulate_dipole(&dip, pos.data(), ad, base.rvdw, latt, cell.volume, cfg.mbd_beta, cfg.mbd_a,
                      true, false, nullptr);
    Mat tgg(static_cast<int>(3 * n));
    for (std::size_t i = 0; i < n; ++i) {
      for (int a = 0; a < 3; ++a) {
        tgg(3 * static_cast<int>(i) + a, 3 * static_cast<int>(i) + a) = 1.0 / ad[i];
      }
    }
    for (int i = 0; i < static_cast<int>(3 * n); ++i) {
      for (int j = 0; j < static_cast<int>(3 * n); ++j) {
        tgg(i, j) += dip.re[static_cast<std::size_t>(i * static_cast<int>(3 * n) + j)];
      }
    }
    Mat inv(0);
    if (!invert(tgg, &inv)) {
      throw std::runtime_error("MBD screening matrix is singular");
    }
    for (std::size_t i = 0; i < n; ++i) {
      double tr = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (std::size_t j = 0; j < n; ++j) {
          tr += inv(3 * static_cast<int>(i) + a, 3 * static_cast<int>(j) + a);
        }
      }
      alpha_w[static_cast<std::size_t>(f) * n + i] = tr / 3.0;
    }
  }
  std::vector<double> alpha(n), c6(n), rvdw(n), omega(n);
  for (std::size_t i = 0; i < n; ++i) {
    alpha[i] = alpha_w[i];
    double acc = 0.0;
    for (int f = 0; f < kMbdNfreq; ++f) {
      const double aw = alpha_w[static_cast<std::size_t>(f) * n + i];
      acc += kMbdWeight[f] * aw * aw;
    }
    c6[i] = 3.0 / kPi * acc;
    rvdw[i] = base.rvdw[i] * std::cbrt(alpha[i] / base.alpha[i]);
    omega[i] = (4.0 / 3.0) * c6[i] / (alpha[i] * alpha[i]);
  }
  const int nk0 = cfg.mbd_gamma ? 1 : cfg.mbd_k[0];
  const int nk1 = cfg.mbd_gamma ? 1 : cfg.mbd_k[1];
  const int nk2 = cfg.mbd_gamma ? 1 : cfg.mbd_k[2];
  const int nk = nk0 * nk1 * nk2;
  double ene = 0.0;
  for (int iz = 0; iz < nk2; ++iz) {
    for (int iy = 0; iy < nk1; ++iy) {
      for (int ix = 0; ix < nk0; ++ix) {
        double frac[3] = {0.0, 0.0, 0.0};
        if (!cfg.mbd_gamma) {
          frac[0] = (ix + 0.5) / nk0;
          frac[1] = (iy + 0.5) / nk1;
          frac[2] = (iz + 0.5) / nk2;
          for (int c = 0; c < 3; ++c) {
            if (frac[c] > 0.5) {
              frac[c] -= 1.0;
            }
          }
        }
        double kpt[3] = {};
        for (int c = 0; c < 3; ++c) {
          kpt[c] = 2.0 * kPi * (frac[0] * cell.recip[0][c] + frac[1] * cell.recip[1][c] +
                                frac[2] * cell.recip[2][c]);
        }
        DipoleWork dip(static_cast<int>(n));
        accumulate_dipole(&dip, pos.data(), alpha, rvdw, latt, cell.volume, cfg.mbd_beta, cfg.mbd_a,
                          false, true, kpt);
        const int m = static_cast<int>(3 * n);
        Mat doubled(2 * m);
        for (int p = 0; p < m; ++p) {
          for (int q = 0; q < m; ++q) {
            double ar = dip.re[static_cast<std::size_t>(p * m + q)];
            double ai = dip.im[static_cast<std::size_t>(p * m + q)];
            const int ip = p / 3;
            const int jp = q / 3;
            const double pre = omega[static_cast<std::size_t>(ip)] *
                               std::sqrt(alpha[static_cast<std::size_t>(ip)]) *
                               omega[static_cast<std::size_t>(jp)] *
                               std::sqrt(alpha[static_cast<std::size_t>(jp)]);
            ar *= pre;
            ai *= pre;
            if (p == q) {
              ar += omega[static_cast<std::size_t>(ip)] * omega[static_cast<std::size_t>(ip)];
            }
            doubled(p, q) += ar;
            doubled(p + m, q + m) += ar;
            doubled(p, q + m) += -ai;
            doubled(p + m, q) += ai;
          }
        }
        for (int p = 0; p < 2 * m; ++p) {
          for (int q = p + 1; q < 2 * m; ++q) {
            const double s = 0.5 * (doubled(p, q) + doubled(q, p));
            doubled(p, q) = s;
            doubled(q, p) = s;
          }
        }
        std::vector<double> eval;
        jacobi(doubled, &eval);
        for (double lam : eval) {
          if (lam < -1e-6) {
            throw std::runtime_error("MBD eigenvalue is negative");
          }
          ene += std::sqrt(std::max(lam, 0.0)) / 4.0;
        }
      }
    }
  }
  ene /= static_cast<double>(nk);
  double wsum = 0.0;
  for (double w : omega) {
    wsum += w;
  }
  return ene - 1.5 * wsum;
}
