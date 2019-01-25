import subprocess
import time
import numpy             as np
import numpy.ctypeslib   as ctl
import ctypes            as ct
import matplotlib.pyplot as plt

s0 = 0

nparticles = 100000
nt = 100

nb = 50 #int(5 * np.sqrt(nt))
na = -nb

nx = nb - na + 1

subprocess.run(["make"])
lib = ctl.load_library("lib.so", "./")

class Complex(ct.Structure):
    _fields_ = [ ("real", ct.c_double)
               , ("imag", ct.c_double)]

lib.c_kernel_feynman.restype = ct.POINTER(Complex)
t1 = time.perf_counter()
p = lib.c_kernel_feynman(s0, nparticles, nt, na, nb)
t2 = time.perf_counter()
print(str(t2 - t1) + " s")

ys = ctl.as_array(p, shape = (nx,))
xs = np.array([na + i for i in range(nx)])

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
plt.legend()

# particle distribution from Central limit theorem
#ys_clt = nparticles * np.exp(-xs**2 / (2 * nt * 4)) / np.sqrt(2 * np.pi * nt * 4)
ys_clt = rys[na] * np.exp(-xs**2 / (2 * nt * 4))
plt.plot(xs, ys_clt, '-', label = 'CLT\'')

plt.savefig("feynman.png")
