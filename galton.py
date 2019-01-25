import subprocess
import numpy             as np
import numpy.ctypeslib   as ctl
import ctypes            as ct
import matplotlib.pyplot as plt

s0 = 1

nparticles = 40000
nt = 50

nb = int(3 * np.sqrt(nt))
na = -nb

nx = nb - na + 1

subprocess.run(["make"])
lib = ctl.load_library("lib.so", "./")

#######################
# simple Galton board #
#######################

lib.c_kernel_galton_simple.restype = ct.POINTER(ct.c_double)
p = lib.c_kernel_galton_simple(s0, nparticles, nt, na, nb)

ys = ctl.as_array(p, shape = (nx,))

xs = np.array([na + i for i in range(nx)])

# from Central limit theorem; normal distribution is exp(-x^2 / 2 sigma^2) / sqrt(2 pi sigma^2)
ys_clt = 2 * nparticles * np.exp(-xs**2 / (2 * nt)) / np.sqrt(2 * np.pi * nt)

plt.plot(xs, ys, '.', label = 'MC')
plt.plot(xs, ys_clt, '-', label = 'CLT')
plt.legend()

#########################
# advanced Galton board #
#########################

lib.c_kernel_galton.restype = ct.POINTER(ct.c_double)
p = lib.c_kernel_galton(s0, nparticles, nt, na, nb)

ys = ctl.as_array(p, shape = (nx,))

xs = np.array([na + i for i in range(nx)])

# from Central limit theorem; normal distribution is exp(-x^2 / 2 sigma^2) / sqrt(2 pi sigma^2)
# a = 3
# ys_clt = nparticles * np.exp(-xs**2 / (2 * nt * 4/3)) / np.sqrt(2 * np.pi * nt * 4/3)
# a = 5
ys_clt = nparticles * np.exp(-xs**2 / (2 * nt * 4)) / np.sqrt(2 * np.pi * nt * 4)

plt.plot(xs, ys, '.', label = 'MC\'')
plt.plot(xs, ys_clt, '-', label = 'CLT\'')
plt.legend()
plt.savefig("galton.png")
