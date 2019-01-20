import subprocess
import numpy as np
import numpy.ctypeslib as ctl
import ctypes as ct

nparticles = 100
nt = 1

nb = 4 #np.int(np.sqrt(nt))
na = -nb

subprocess.run(["make"])

lib = ctl.load_library("lib.so", "./")

lib.c_kernel_galton.restype = ct.POINTER(ct.c_double)
p = lib.c_kernel_galton(nparticles, nt, na, nb)
a = ctl.as_array(p, shape = (nb - na + 1,))
print(a)
