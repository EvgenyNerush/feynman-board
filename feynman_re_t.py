import subprocess
import time
import numpy             as np
import numpy.ctypeslib   as ctl
import ctypes            as ct
import matplotlib.pyplot as plt

s0 = 51

nparticles = 20000000
nt = 6

no = 100
nb = 1000

nx = 2 * no + 1

subprocess.run(["make"])
lib = ctl.load_library("lib.so", "./")

class Complex(ct.Structure):
    _fields_ = [ ("real", ct.c_double)
               , ("imag", ct.c_double)]

lib.c_kernel_feynman_re_t.restype = ct.POINTER(Complex)
t1 = time.perf_counter()
p = lib.c_kernel_feynman_re_t(s0, nparticles, nt, no, nb)
t2 = time.perf_counter()
print(str(t2 - t1) + " s")

ys = ctl.as_array(p, shape = (nx,))
xs = np.array([-no + i for i in range(nx)])

def re(a):
    x, _ = a
    return x

def im(a):
    _, y = a
    return y

rys = np.array(list(map(re, ys)))
iys = np.array(list(map(im, ys)))

plt.plot(xs, rys, '-', label = 'MC, Re')
#plt.plot(xs, iys, '-', label = 'MC, Im')

# from Central limit theorem; normal distribution is exp(-x^2 / 2 sigma^2) / sqrt(2 pi sigma^2)
ampl = sum(np.abs(rys[no:(no + 10)])) / 10
ys_clt = ampl * np.exp(-xs**2 / (2 * nt * 24**2))
#ys_estim = -ampl * np.cos(-0.5 * np.pi * xs**2 / (7 * nt**2))
ys_estim = ys_clt * np.cos(-0.25 * np.pi * xs**2 / (7 * nt**2))
#plt.plot(xs, ys_clt, '--', label = 'CLT')
plt.plot(xs, ys_estim, '-', label = 'est')

plt.legend()

plt.savefig("feynman_re_t.png")
