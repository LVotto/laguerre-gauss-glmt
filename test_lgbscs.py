# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:17:38 2018

@author: luiz_
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma, gammasgn

import lgbscs
import spc

def test_lgs():
    g = []
    ns = np.arange(250, 300)
    for n in ns:
        g.append(lgbscs.lg_bsc_vplus1(n, 0, s=.001))
    plt.plot(ns, np.real(g))
    plt.show()
    plt.plot(ns, np.imag(g))
    plt.show()

def gamma(n, v, m):
    nums = [(n - v + 1) / 2, 1 / 2 + n - m]
    dens = [m + 1, (n - v) / 2, (n - 2 * m + 1) / 2]
    return spc.gamma_quotients(nums, dens)

def test_gamma(nmin, nmax):
    g = []
    ns = np.arange(nmin, nmax)
    for n in ns:
        g.append(gamma(n, 0, 0))
    plt.plot(ns, g)
    plt.show()
    
def loggamma_quotients(num_args=[], den_args=[]):
    log = 0
    sgn = 1
    for arg in (num_args + den_args):
        sgn *= gammasgn(arg)
    for num in num_args:
        log += loggamma(num)
    for den in den_args:
        log -= loggamma(den)
    return sgn, log

def log_exponentials(kwargs):
    log = 0
    for base in kwargs:
        log += kwargs[base] * np.log(base)
    return log

def eval_special_exp(gamma_args={}, exp_args={}):
    sgn, lgamma = loggamma_quotients(**gamma_args)
    logexp = log_exponentials(exp_args)
    return sgn * np.exp(lgamma + logexp)
    #return lgamma + logexp

def difficult_expression(n, m, v, s=.001):
    num_args = [(n - v + 1) / 2, n - m + .5]
    den_args = [(n + v + 2) / 2, m + 1, (n - 2 * m - v + 1) / 2]
    exp_kwargs = {2: 7 / 2 + n - 2 * m,
                  s: n - 2 * m + 2}
    r = eval_special_exp(gamma_args={'num_args': num_args, 'den_args': den_args},
                         exp_args = exp_kwargs)
    return r

def test_difficult_exp(nmax, m, v, s=.001):
    ns = np.arange(10, nmax, 2)
    p = []
    for n in ns:
        p.append(difficult_expression(n, m, v, s=s))
    plt.plot(ns, p)
    plt.show()
    return p
    
if __name__ == '__main__':
    p = test_difficult_exp(1000, 500, 0);