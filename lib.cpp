#include <cstdint>
#include <vector>
#include <complex>
#include <iostream>
#include <cmath>

using namespace std;

/* Kernel (Green's function) of something similar to Schroedinger equation, computed with Monte
 * Carlo path integration and Metropolis algorithm.
 */
template<typename T> vector<T> kernel
    ( uint32_t s0         // to seed RNGs
    , long np             // number of particles to use
    , long nt             // number of timesteps
    , long na             // left x boundary (including)
    , long nb             // right x boundary (including)
    , long w(uint32_t&)   // RNG that produces numbers wich have the  distribution of
                          // beans scattered by a single nail
    , T u(long)           // the multiplier produced on a step, as function of the
                          // displacement; for the Galton board u(_) = 1
    ) {
        vector<long> x(np, 0); // initially x = 0
        vector<T> m(np, 1); // to store exp(-i S / hbar)

        vector<uint32_t> s(np); // states for RNG
        for (long i = 0; i < np; ++i) {
            // to avoid zero and produce completely new series for s0 + 1
            s[i] = s0 * static_cast<uint32_t>(np) + static_cast<uint32_t>(i + 1);
            // "heating" of RNG to produce uniform distribution of initial seeds; e.g., one step is
            // enough for Park--Miller RNG
            w(s[i]);
        }

        // sum along the trajectories
        # pragma omp parallel for
        for (long i = 0; i < np; ++i) {
            for (long j = 0; j < nt; ++j) {
                long delta_x = w(s[i]);
                x[i] += delta_x;
                m[i] *= u(delta_x);
            }
        }

        // sum for the trajectories which end in the same position
        long nx = nb - na + 1;
        vector<T> res(nx, 0);
        for (long i = 0; i < np; ++i) {
            if (x[i] >= na && x[i] <= nb) {
                size_t j = static_cast<size_t>(x[i] - na);
                res[j] += m[i];
            }
        }

        return res;
}

// Park--Miller RNG
const uint64_t pm_randmax = 0x7FFFFFFFull;
void pm_rng(uint32_t& state) {
    state = static_cast<uint32_t>
        (48271ull * static_cast<uint64_t>(state) % pm_randmax);
}

// delta x = +-1
long w_galton_simple(uint32_t& state) {
    pm_rng(state);
    if (state > static_cast<uint32_t>(pm_randmax >> 1)) {
        return -1;
    } else {
        return 1;
    }
}


// symmetric "triangle" distribution from -a to a. The interval (0, randmax) can be devided on a^2
// subintervals each of which corresponds to some number in (-a, a).  Namely, (a) of such intervals
// should yield 0, (a - 1) should yield 1, etc.
long w_galton(uint32_t& state) {
    pm_rng(state);
    const uint64_t a = 20; // actually it is assumed that *a* is small

    // ascending part of the distribution, including the central point
    uint64_t j = 0;
    for (uint64_t i = 1; i <= a; ++i) {
        if (  state >= pm_randmax * j / (a * a)
           && state < pm_randmax * (j + i) / (a * a)) {
            return -static_cast<long>(a - i);
        }
        j += i;
    }
    // descending part
    for (uint64_t i = a - 1; i > 0; --i) {
        if (  state >= pm_randmax * j / (a * a)
           && state < pm_randmax * (j + i) / (a * a)) {
            return static_cast<long>(a - i);
        }
        j += i;
    }
    return 0; // to avoid warning, never happens
}

double u_galton(long _) {
    return 1;
}

complex<double> u_feynman(long x) {
    complex<double> i;
    i.real(0);
    i.imag(1);
    //return exp(-i * 0.5 * M_PI * pow(static_cast<double>(x) / 10.0, 2));
    return exp(-pow(static_cast<double>(x) / 3.0, 2));
}

// Interface //

template<typename T> T* v_copy(vector<T> v) {
    T* p = new T[v.size()];
    for (size_t i = 0; i < v.size(); ++i) {
        p[i] = v[i];
    }
    return p;
}

extern "C" {
    double* c_kernel_galton_simple(uint32_t s0, long np, long nt, long na, long nb) {
        vector<double> v = kernel<double>(s0, np, nt, na, nb, w_galton_simple, u_galton);
        return v_copy(v);
    }

    double* c_kernel_galton(uint32_t s0, long np, long nt, long na, long nb) {
        vector<double> v = kernel<double>(s0, np, nt, na, nb, w_galton, u_galton);
        return v_copy(v);
    }

    complex<double>* c_kernel_feynman(uint32_t s0, long np, long nt, long na, long nb) {
        vector<complex<double>> v =
            kernel<complex<double>>(s0, np, nt, na, nb, w_galton, u_feynman);
        return v_copy(v);
    }
}
