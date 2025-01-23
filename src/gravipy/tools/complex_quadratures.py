import numpy as np

def mathfunc_real(values, dt):
    return np.trapz(np.real(values), dx=dt, axis=0)

def mathfunc_imag(values, dt):
    return np.trapz(np.imag(values), dx=dt, axis=0)

def complex_quadrature_num(func, a, b, theta, nsteps=int(1e2)):
    t = np.logspace(np.log10(a), np.log10(b), nsteps)
    dt = np.diff(t, axis=0)
    values = func(t, *theta)
    real_integral = mathfunc_real(values, dt)
    imag_integral = mathfunc_imag(values, dt)
    return real_integral + 1j*imag_integral
