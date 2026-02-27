# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 23:57:11 2026

@author: khate pontay
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

#LIKELIHOOD
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()