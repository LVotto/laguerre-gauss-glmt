# -*- coding: utf-8 -*-
"""
Special functions

Created on Fri May 25 01:23:04 2018

@author: luiz_
"""

import scipy.special as spc
import numpy as np

MIE_ARGS = ['diameter', 'wavelength', 'index', 'mu_sp', 'mu']
BSC_ARGS = ['v', 'rho', 'phi', 'z',
            'coeffs', 'axicons', 'kzs', 'method']

def hankel(order, argument, kind=2):
    sign = 1 if kind == 1 else -1
    return spc.jv(order, argument) + sign * 1j * spc.yv(order, argument)

def spherical_jn(order, argument):
    """" Reimplementation of Spherical Bessel Function of first kind as
         scipy crashes for complex arguments on Python 3.6. 
    """
    return np.sqrt(np.pi / (2 * argument)) * spc.jv(order + 1 / 2, argument)

def spherical_yn(order, argument):
    """" Reimplementation of Spherical Bessel Function of second kind as
         scipy crashes for complex arguments on Python 3.6. 
    """
    return np.sqrt(np.pi / (2 * argument)) * spc.yv(order + 1 / 2, argument)

def spherical_hankel2(order, argument):
    return (spherical_jn(order, argument) \
            - 1j * spherical_yn(order, argument))

def riccati_bessel(order, argument, kind=1, overflow_protection=True):
    if overflow_protection:
        if kind == 1:
            return np.sqrt(np.pi * argument / 2) * spc.jv(order + 1 / 2, argument)
        elif kind == 2:
            return (np.sqrt(np.pi * argument / 2) \
                   * (spc.jv(order + 1 / 2, argument) \
                   - 1j * spc.yv(order + 1 / 2, argument)))
        elif kind == 3:
            return np.sqrt(np.pi * argument / 2) * spc.yv(order + 1 / 2, argument)
        else:
            raise ValueError("Only kind=1 and kind=2 allowed.")
    
    if kind == 1:
        return argument * spherical_jn(order, argument)
    elif kind == 2:
        return argument * spherical_hankel2(order, argument)
    elif kind == 3:
        return argument * spherical_yn(order, argument)
    else:
        raise ValueError("Only kind=1 and kind=2 allowed.")

def d_riccati_bessel(order, argument, kind=1):
    if kind == 1:
        sp_function = spherical_jn
    elif kind == 2:
        sp_function = spherical_hankel2
    elif kind == 3:
        sp_function = spherical_yn
    else:
        raise ValueError("Only kind=1 and kind=2 allowed.")
    
    return (argument * sp_function(order - 1, argument) \
            - (order) * sp_function(order, argument))
    
def riccati_semi_wronskian1(n, x, M, f_bessel=[spc.jv, spc.jv]):
    result = (x * f_bessel[0](n - 1 / 2, x) - n * f_bessel[0](n + 1 / 2, x))
    return result * np.sqrt(M) * np.pi / 2 * f_bessel[1](n + 1 / 2, M * x)
    
def riccati_semi_wronskian2(n, x, M, f_bessel=[spc.jv, spc.jv]):
    return riccati_semi_wronskian1(n, M * x, 1 / M,
                                   f_bessel=[f_bessel[1], f_bessel[0]])
    
def mie_coeff_a(n, diameter=35E-6, wavelength=1064E-9, index=1.01,
                mu_sp=1, mu=1, overflow_protection=True):
    
    a = np.pi * diameter / wavelength
    b = index * a
    numerator = (mu_sp * riccati_bessel(n, a) * d_riccati_bessel(n, b) \
                 - mu * index * d_riccati_bessel(n, a) * riccati_bessel(n, b))
    denominator = (mu_sp * riccati_bessel(n, a, kind=2) \
                   * d_riccati_bessel(n, b) \
                   - mu * index * d_riccati_bessel(n, a, kind=2) \
                   * riccati_bessel(n, b))
    return numerator / denominator

def mie_coeff_a_alt(n, diameter=35E-6, wavelength=1064E-9, index=1.01,
                    mu_sp=1, mu=1, overflow_protection=True):
    a = np.pi * diameter / wavelength
    b = index * a
    
    if overflow_protection:
        fa = (mu_sp * riccati_semi_wronskian2(n, a, index) \
              - index * mu * riccati_semi_wronskian1(n, a, index))
        bess = [spc.yv, spc.jv]
        fb = (mu_sp * riccati_semi_wronskian2(n, a, index, f_bessel=bess) \
              - index * mu * riccati_semi_wronskian1(n, a, index,
                                                     f_bessel=bess))
        if fb == 0:
            return 1
        
        return fa * (fa - 1j * fb) / (fa ** 2 + fb ** 2)

    fa = (mu_sp * riccati_bessel(n, a) * d_riccati_bessel(n, b) \
          - mu * index * d_riccati_bessel(n, a) * riccati_bessel(n, b))
    fb = (mu_sp * riccati_bessel(n, a, kind=3) \
           * d_riccati_bessel(n, b) \
           - mu * index * d_riccati_bessel(n, a, kind=3) \
           * riccati_bessel(n, b))
    if fb == 0:
        return 1
    
    return fa * (fa + 1j * fb) / (fa ** 2 + fb ** 2)

def mie_coeff_b(n, diameter=35E-6, wavelength=1064E-9, index=1.01,
                mu_sp=1, mu=1, overflow_protection=True):
    a = np.pi * diameter / wavelength
    b = index * a
    if overflow_protection:
        return mie_coeff_b_alt(n, diameter=diameter,
                               wavelength=wavelength, index=index,
                               mu_sp=mu_sp, mu=mu, overflow_protection=True)
    
    numerator = (mu * index * riccati_bessel(n, a) * d_riccati_bessel(n, b) \
                 - mu_sp * d_riccati_bessel(n, a) * riccati_bessel(n, b))
    denominator = (mu * index * riccati_bessel(n, a, kind=2) \
                   * d_riccati_bessel(n, b) \
                   - mu_sp * index * d_riccati_bessel(n, a, kind=2) \
                   * riccati_bessel(n, b))
    return numerator / denominator

def mie_coeff_b_alt(n, diameter=35E-6, wavelength=1064E-9, index=1.01,
                    mu_sp=1, mu=1, overflow_protection=True):
    a = np.pi * diameter / wavelength
    b = index * a
    
    if overflow_protection:
        fa = (index * mu * riccati_semi_wronskian2(n, a, index) \
              - mu_sp * riccati_semi_wronskian1(n, a, index))
        bess = [spc.yv, spc.jv]
        fb = (index * mu * riccati_semi_wronskian2(n, a, index, f_bessel=bess) \
              - mu_sp * riccati_semi_wronskian1(n, a, index,
                                                     f_bessel=bess))
        if fb == 0:
            return 1
        
        return fa * (fa + 1j * fb) / (fa ** 2 + fb ** 2)
    
    fa = (mu * index * riccati_bessel(n, a) * d_riccati_bessel(n, b) \
                 - mu_sp * d_riccati_bessel(n, a) * riccati_bessel(n, b))
    fb = (mu * index * riccati_bessel(n, a, kind=3) \
           * d_riccati_bessel(n, b) \
           - mu_sp * index * d_riccati_bessel(n, a, kind=3) \
           * riccati_bessel(n, b))
    return fa * (fa + 1j * fb) / (fa ** 2 + fb ** 2)

def scatt_eff(x, wavelength=1064E-9, index=1.01,
              mu_sp=1, mu=1):
    result = 0
    kwargs = {'diameter': x, 'wavelength': wavelength, 'index': index,
              'mu_sp': mu_sp, 'mu': mu}
    for n in range(1, get_max_it(x, wave_number_k=1) + 160):
        increment = (2 * n + 1) * (np.abs(mie_coeff_a_alt(n, **kwargs) ** 2) \
                                   * np.abs(mie_coeff_b_alt(n, **kwargs)) ** 2)
        result += increment
    return (2 / x ** 2) * result

def fac_plus_minus(n, m):
    """ Calculates the expression below avoiding overflows.
    
    .. math::
        \\frac{(n + m)!}{(n - m)!}
    """
    product = 1
    if n > 0:
        if m > 0:
            if n - m >= 0:
                for factor in range(n - m + 1, n + m + 1):
                    product *= factor
                return product
            else:
                for factor in range(m - n + 1, m + n + 1):
                    product *= factor
                return pow(-1, n - m) * product
        if m == 0:
            return 1

def get_max_it(x_max, wave_number_k):
    """ Calculates stop iteration number """
    max_it = np.ceil(wave_number_k * x_max + np.longdouble(4.05) \
                     * pow(wave_number_k * x_max, 1/3)) + 2
    if np.isnan(max_it):
        return 2
    return int(max_it)
            

def gamma_quotient(num_arg, den_arg):
    return np.exp(spc.loggamma(num_arg) - spc.loggamma(den_arg))

def gamma_quotients(num_args=[], den_args=[]):
    import warnings
    warnings.filterwarnings("error")
    log = 0
    sgn = 1
    for arg in (num_args + den_args):
        sgn *= spc.gammasgn(arg)
    for num in num_args:
        log += spc.loggamma(num)
    for den in den_args:
        log -= spc.loggamma(den)
    try:
        res = sgn * np.exp(log)
        warnings.filterwarnings("default")
        return res
    except RuntimeWarning:
        with open('./runtimewarnings.txt', 'a') as f:
            f.writelines(str([num_args, den_args]) + '\n')
        warnings.filterwarnings("default")
        return sgn * np.exp(log)

def exponentials(kwargs):
    log = 0
    for base in kwargs:
        log += kwargs[base] * np.log(base)
    return np.exp(log)
        
def partial_factorial(n, m):
    """ Computes n! / (n - m)! """
    factorial = 1
    for p in range(n - m + 1, n):
        factorial *= p
    return factorial

def gouesbet_epsilon(n, u):
    return 1 if n > u else 0
                
                