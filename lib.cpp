#include <cstdint>
#include <vector>
#include <iostream>

using namespace std;

/* Kernel (Green's function) of something similar to Schroedinger equation, computed with Monte
 * Carlo path integration and Metropolis algorithm.
 */
template<typename T> vector<T> kernel
    ( long np             // number of particles to use
    , long nt             // number of timesteps
    , long na             // left x boundary (including)
    , long nb             // right x boundary (including)
    , long w(uint32_t&)   // RNG that produces numbers wich have the  distribution of
                          // beans scattered by a single nail
    , T u(long)           // the multiplier produced on a step, as function of the
                          // displacement; for the Galton board u(_) = 1
    ) {
        vector<long> x(np, 0); // initially x = 0
        vector<long> m(np, 1); // to store exp(-i S / hbar)

        vector<uint32_t> s(np); // states for RNG
        for (long i = 0; i < np; ++i) {
            s[i] = 0xffffffff - i;
        }

        // sum along the trajectories
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

long w_galton(uint32_t& state) {
    state = static_cast<uint32_t>(48271ull * static_cast<uint64_t>(state) % 0x7FFFFFFFull); // Park--Miller RNG
    if (state > 0x40000000ul) {
        return -1;
    } else {
        return 1;
    }
}

double u_galton(long _) {
    return 1;
}

////////////////////////////////////////

extern "C" {
    double* c_kernel_galton(long np, long nt, long na, long nb) {
        vector<double> v = kernel<double>(np, nt, na, nb, w_galton, u_galton);
        double* p = new double[v.size()];
        for (size_t i = 0; i < v.size(); ++i) {
            p[i] = v[i];
        }
        return p;
    }
}
