import subprocess
import time
import numpy             as np
import numpy.ctypeslib   as ctl
import ctypes            as ct
import matplotlib.pyplot as plt

s0 = 3

nparticles = 20000000
nt = 7

no = 40 #int(5 * np.sqrt(nt))
nb = 15

nx = 2 * no + 1

subprocess.run(["make"])
lib = ctl.load_library("lib.so", "./")

class Complex(ct.Structure):
    _fields_ = [ ("real", ct.c_double)
               , ("imag", ct.c_double)]

lib.c_kernel_feynman_img_t.restype = ct.POINTER(Complex)
t1 = time.perf_counter()
p = lib.c_kernel_feynman_img_t(s0, nparticles, nt, no, nb)
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
plt.plot(xs, iys, '-', label = 'MC, Im')

# particle distribution from Central limit theorem
sigma_w = 8.15
#ys_clt = nparticles * np.exp(-xs**2 / (2 * nt * sigma_w**2)) / np.sqrt(2 * np.pi * nt * sigma_w)
ys_clt = rys[no] * np.exp(-xs**2 / (2 * nt * 8.15**2))
plt.plot(xs, ys_clt, '-', label = 'CLT\'')

# just an estimate
ys_theory = rys[no] * np.exp(-xs**2 / (0.5 * nt**2 * 3))
plt.plot(xs, ys_theory, '-', label = 'estim')
plt.legend()

plt.savefig("feynman_img_t.png")
