# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:16:35 2018

@author: luiz_
"""

import numpy as np

import spc

def lg_bsc_vplus1(n, v, mu=0, s=.2):
    if mu != 0:
        raise NotImplementedError('Only mu = 0 is implemented yet.')
    # AS CALCULATED IN SHEET I
    if n < v:
        return 0
    g = 0
    m = 0
    
    if (n - v) % 2:
        while m <= n / 2 and (n - 2 * m) > v:
            num_args = [(n - v + 1) / 2, n - m + .5]
            den_args = [(n + v) / 2 + 1, m + 1, (n - v + 1) / 2 - m]
            gamma = spc.gamma_quotients(num_args=num_args, den_args=den_args)
            exps = spc.exponentials({2: n - 2 * m - v / 2 - 2,
                                     s: n - 2 * m})
            g += np.exp(1j * np.pi * (3 * n - 2 * m + 1) / 2) * exps * gamma
            m += 1        
    else:
        while m <= n / 2 and (n - 2 * m) > (v + 1):
            num_args = [(n - v) / 2, n - m + .5]
            _den_args = [(m + 1), (n + v + 3) / 2]
            exps = spc.exponentials({2: n - 2 * m - v / 2 - 3,
                                     s: n - 2 * m - 1})
            den_args2 = _den_args[:]
            den_args2.append((n - 2 * m - v) / 2)
            gamma2 = spc.gamma_quotients(num_args=num_args, den_args=den_args2)
            gamma2 *= (2 * s ** 2 * (v + 1) - 1)
            
            gamma1 = 0
            if (n - 2 * m) > (v + 3):
                den_args1 = _den_args[:]
                den_args1.append((n - 2 * m - v + 1) / 2)
                gamma1 = spc.gamma_quotients(num_args=num_args,
                                             den_args=den_args1)
                gamma1 *= -2j * np.power(s, 5)
            
            g += (np.exp(1j * np.pi * (3 * n - 2 * m) / 2) * exps \
                  * (gamma1 + gamma2))
            m += 1                
    return g           

def lg_bsc_vminus1(n, v, mu=0, s=.2):
    # AS CALCULATED IN SHEET II
    if (n - v) % 2:
        return -4 * (n + v + 2) * lg_bsc_vplus1(n, v, mu=mu, s=s) / (n - v - 1)
    
    return -4 * (n + v + 3) * lg_bsc_vplus1(n, v, mu=mu, s=s) / (n - v)